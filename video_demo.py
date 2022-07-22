from easydict import EasyDict

import tensorflow.compat.v1 as tf
import numpy as np
import torch
import clip
import cv2
import webcolors
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

single_template = text_encoder.single_template
multiple_templates = text_encoder.multiple_templates

max_boxes_to_draw = 25
nms_threshold = 0.1
min_rpn_score_thresh = 0.9
min_box_area = 50
conf_threshold = 0.4

clip.available_models()
model, preprocess = clip.load("ViT-B/32")


session = tf.Session(graph=tf.Graph())
saved_model_dir = './save_models' #@param {type:"string"}
_ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)


def map_cat_to_color(num_categories):
  color_map = [util.STANDARD_COLORS[(i*i + num_categories) % len(util.STANDARD_COLORS)] for i in range(num_categories)]
  return [webcolors.name_to_rgb(color) for color in color_map]

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

    indices = np.argsort(-np.max(scores_all, axis=1))  # Results are ranked by scores
    indices_fg = np.array([i for i in indices if np.argmax(scores_all[i]) != 0])


    #################################################################
    # Plot detected boxes on the input image.
    ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)


    ######################################
    # MY OWN CODE WILL GO HERE
    img = cv2.imread(image_path,  cv2.COLOR_BGR2RGB)
    if len(indices_fg) == 0:
      print('ViLD does not detect anything belong to the given category')

    else:
      boxes = rescaled_detection_boxes[indices_fg]
      probs = detection_roi_scores[indices_fg]
      n_boxes = boxes.shape[0]
      color_map = map_cat_to_color(len(category_names))
      
      for box, anno_idx, prob in zip(boxes, indices[0:int(n_boxes)], probs):
        
        scores = scores_all[anno_idx]
        
        if np.argmax(scores) == 0:
          continue

        if(np.max(scores) < 0.9):
          continue

        ymin, xmin, ymax, xmax = box
        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        thickness = 2
        class_id = np.argmax(scores)
        color = color_map[class_id]

        #print(category_names[class_id], " ", prob, " ", np.max(scores))

        cv2.rectangle(img, start_point, end_point, color, thickness)
        cv2.putText(img, category_names[class_id], (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)


    return img, image_height, image_width

def main(scene = 'scene_04', test_mode = False, additional_labels = []):
  image_dir = os.path.join('/datasets/kitti/scenes/', scene)
  img_list = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
  img_list.sort()

  category_names = [ 
      "Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"
  ]
  category_names = ['background'] + category_names + additional_labels
  print(category_names)
  categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(category_names)]

  # Compute text embeddings and detection scores, and rank results
  text_features = build_text_embedding(categories, model, FLAGS)

  video_array = []

  if test_mode == False:
    size = ()
    for image in tqdm(img_list):
      adapted_img, height, width= inference(image, category_names, text_features)
      video_array.append(adapted_img)
      size = (width, height)

    out = cv2.VideoWriter('test.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

    for i in range(len(video_array)):
        out.write(video_array[i])
    out.release()

  else:
    image = img_list[0]
    result, _, _ = inference(image, category_names, text_features)
    cv2.imwrite("test.png", result)

if __name__ == "__main__":
  scene = 'scene_04'
  additional_labels = []
  #additional_labels = ['Traffic Light', 'Traffic Sign', 'Road Light', 'Lane Markings', 'Licences Plate']
  main(scene, test_mode = False, additional_labels = additional_labels)