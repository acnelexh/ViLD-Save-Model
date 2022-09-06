from distutils.command.config import config
from easydict import EasyDict

import tensorflow.compat.v1 as tf
import numpy as np
import torch
import clip
import cv2
import text_encoder
import util

from scipy.special import softmax
from tqdm import tqdm
from PIL import Image

from text_encoder import build_text_embedding, article, processed_name

import json
import os



#Global Config, Change this later
FLAGS = {
    'prompt_engineering': True,
    'this_is': True,
    
    'temperature': 100.0,
    'use_softmax': True,
    'run_on_gpu' : torch.cuda.is_available()
}
FLAGS = EasyDict(FLAGS)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)


single_template = text_encoder.single_template
multiple_templates = text_encoder.multiple_templates

max_boxes_to_draw = 25
nms_threshold = 0.3
min_rpn_score_thresh = 0.9
min_box_area = 100
conf_threshold = 0.7


clip.available_models()
model, preprocess = clip.load("ViT-B/32")


session = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options))
saved_model_dir = './save_models' #@param {type:"string"}
_ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)

def inference(image_path, category_names, text_features):
    #################################################################
    # Obtain results and read image
    roi_boxes, roi_scores, detection_boxes, scores_unused, box_outputs, detection_masks, visual_features, image_info = session.run(
            ['RoiBoxes:0', 'RoiScores:0', '2ndStageBoxes:0', '2ndStageScoresUnused:0', 'BoxOutputs:0', 'MaskOutputs:0', 'VisualFeatOutputs:0', 'ImageInfo:0'],
            feed_dict={'Placeholder:0': [image_path,]})


    roi_boxes = np.squeeze(roi_boxes, axis=0)  # squeeze
    # no need to clip the boxes, already done
    roi_scores = np.squeeze(roi_scores, axis=0)


    detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
    scores_unused = np.squeeze(scores_unused, axis=0)
    box_outputs = np.squeeze(box_outputs, axis=0)
    visual_features = np.squeeze(visual_features, axis=0)

    image_info = np.squeeze(image_info, axis=0)  # obtain image info
    image_scale = np.tile(image_info[2:3, :], (1, 2))
    image_height = int(image_info[0, 0])
    image_width = int(image_info[0, 1])

    rescaled_detection_boxes = detection_boxes / image_scale # rescale

    # Read image
    image = np.asarray(Image.open(open(image_path, 'rb')).convert("RGB"))
    assert image_height == image.shape[0]
    assert image_width == image.shape[1]


    #################################################################
    # Filter boxes

    # Apply non-maximum suppression to detected boxes with nms threshold.
    nmsed_indices = util.nms(
        detection_boxes,
        roi_scores,
        thresh=nms_threshold
        )


    # Compute RPN box size.
    box_sizes = (rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]) * (rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])
    

    # Filter out invalid rois (nmsed rois)
    valid_indices = np.where(
        np.logical_and(
            np.isin(np.arange(len(roi_scores), dtype=int), nmsed_indices),
            np.logical_and(
                np.logical_not(np.all(roi_boxes == 0., axis=-1)),
                np.logical_and(
                roi_scores >= min_rpn_score_thresh,
                box_sizes > min_box_area
                )
            )    
        )
    )[0]

    detection_roi_scores = roi_scores[valid_indices][:max_boxes_to_draw, ...]
    detection_boxes = detection_boxes[valid_indices][:max_boxes_to_draw, ...]
    detection_visual_feat = visual_features[valid_indices][:max_boxes_to_draw, ...]
    rescaled_detection_boxes = rescaled_detection_boxes[valid_indices][:max_boxes_to_draw, ...]

    raw_scores = detection_visual_feat.dot(text_features.T)


    if FLAGS.use_softmax:
      scores_all = softmax(FLAGS.temperature * raw_scores, axis=-1)
    else:
      scores_all = raw_scores
    

    #################################################################
    # Plot detected boxes on the input image.
    indices = np.argsort(-np.max(scores_all, axis=1))  # Results are ranked by scores
    indices_fg = np.array([i for i in indices if np.argmax(scores_all[i]) != 0])

    ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)
    ######################################
    # MY OWN CODE WILL GO HERE
    img = cv2.imread(image_path,  cv2.COLOR_BGR2RGB)
    bbox = []
    bprob = []
    display_labels = []

    if len(indices_fg) == 0:
      print('ViLD does not detect anything belong to the given category')
    else:
      boxes = rescaled_detection_boxes[indices_fg]
      probs = detection_roi_scores[indices_fg]
      n_boxes = boxes.shape[0]
      
      for box, anno_idx, prob in zip(boxes, indices[0:int(n_boxes)], probs):
        
        scores = scores_all[anno_idx]
        
        if np.argmax(scores) == 0:
          continue

        if(np.max(scores) < conf_threshold):
          continue

        bbox.append(box)
        bprob.append(str(np.max(scores))[:4])
        #ymin, xmin, ymax, xmax = box

        class_id = np.argmax(scores)


        display_labels.append(category_names[class_id])



    return img, display_labels, bbox, bprob

def write_result(cat, bbox, prob):
  return f"{cat} 0.00 0 0 {bbox[1]} {bbox[0]} {bbox[3]} {bbox[2]} 0.0 0.0 0.0 0.0 0.0 0.0 0.0 {prob}"

def write_results(file_name, cats, bboxes, probs):
  f = open(file_name, "w")
  output_text = ""
  for cat, bbox, prob in zip(cats, bboxes, probs):
    output_text += write_result(cat, bbox, prob) + "\n"
  f.write(output_text)
  f.close()


def output_file_name(image_name):
  base = os.path.basename(image_name).split('.')[0]

  return base + ".txt"

def main(scene = 'scene_04', additional_labels = []):
  image_dir = '/datasets/kitti/training/image_2'
  eval_dir = '/datasets/kitti/evaluations/base/data'
  img_list = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
  img_list.sort()

  category_names = [ 
      "Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"
  ]
  category_names = ['background'] + category_names + additional_labels
  #print(category_names)
  categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(category_names)]

  # Compute text embeddings and detection scores, and rank results
  text_features = build_text_embedding(categories, model, FLAGS)


  for image in tqdm(img_list):
    file_name = output_file_name(image)
    file_name = os.path.join(eval_dir, file_name)
    _ , display_labels, bboxes, probs = inference(image, category_names, text_features)
    write_results(file_name, display_labels, bboxes, probs)
  #cv2.imwrite(file_name + ".png", result)



if __name__ == "__main__":
  additional_labels = []
  main(additional_labels = additional_labels)




