from easydict import EasyDict

import tensorflow.compat.v1 as tf
import numpy as np
import torch
import clip

from tqdm import tqdm
from PIL import Image


import json
import os



#Global Config, Change this later
FLAGS = {
    'prompt_engineering': True,
    'this_is': True,
    
    'temperature': 100.0,
    'use_softmax': False,
}
FLAGS = EasyDict(FLAGS)


single_template = [
    'a photo of {article} {}.'
]

multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',


    'itap of {article} {}.',
    'itap of my {}.',  # itap: I took a picture of
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',

    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',

    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',

    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',

    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',

    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',

    'a black and white photo of the {}.',
    'a black and white photo of {article} {}.',

    'a plastic {}.',
    'the plastic {}.',

    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',

    'an embroidered {}.',
    'the embroidered {}.',

    'a painting of the {}.',
    'a painting of a {}.',
]



max_boxes_to_draw = 25
nms_threshold = 0.6
min_rpn_score_thresh = 0.9
min_box_area = 220


session = tf.Session(graph=tf.Graph())
saved_model_dir = './save_models' #@param {type:"string"}
_ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)

clip.available_models()
model, preprocess = clip.load("ViT-B/32")


def article(name):
  return 'an' if name[0] in 'aeiou' else 'a'

def processed_name(name, rm_dot=False):
  # _ for lvis
  # / for obj365
  res = name.replace('_', ' ').replace('/', ' or ').lower()
  if rm_dot:
    res = res.rstrip('.')
  return res

def build_text_embedding(categories):
  if FLAGS.prompt_engineering:
    templates = multiple_templates
  else:
    templates = single_template

  run_on_gpu = torch.cuda.is_available()

  with torch.no_grad():
    all_text_embeddings = []
    print('Building text embeddings...')
    for category in tqdm(categories):
      texts = [
        template.format(processed_name(category['name'], rm_dot=True),
                        article=article(category['name']))
        for template in templates]
      if FLAGS.this_is:
        texts = [
                 'This is ' + text if text.startswith('a') or text.startswith('the') else text 
                 for text in texts
                 ]
      texts = clip.tokenize(texts) #tokenize
      if run_on_gpu:
        texts = texts.cuda()
      text_embeddings = model.encode_text(texts) #embed with text encoder
      text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
      text_embedding = text_embeddings.mean(dim=0)
      text_embedding /= text_embedding.norm()
      all_text_embeddings.append(text_embedding)
    all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
    if run_on_gpu:
      all_text_embeddings = all_text_embeddings.cuda()
  return all_text_embeddings.cpu().numpy().T


def nms(dets, scores, thresh, max_dets=1000):
    """Non-maximum suppression.
    Args:
        dets: [N, 4]
        scores: [N,]
        thresh: iou threshold. Float
        max_dets: int.
    """
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0 and len(keep) < max_dets:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        overlap = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-12)

        inds = np.where(overlap <= thresh)[0]
        order = order[inds + 1]
    return keep


def inference(image_path, category_names):
    category_names = ['background'] + category_names
    categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(category_names)]
    category_indices = {cat['id']: cat for cat in categories}


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
    detection_masks = np.squeeze(detection_masks, axis=0)
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
    nmsed_indices = nms(
        detection_boxes,
        roi_scores,
        thresh=nms_threshold
        )

    # Compute RPN box size.
    box_sizes = (rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]) * (rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])
    

    # Filter out invalid rois (nmsed rois)
    valid_indices = np.where(
        np.logical_and(
            np.isin(np.arange(len(roi_scores), dtype=np.int), nmsed_indices),
            np.logical_and(
                np.logical_not(np.all(roi_boxes == 0., axis=-1)),
                np.logical_and(
                roi_scores >= min_rpn_score_thresh,
                box_sizes > min_box_area
                )
            )    
        )
    )[0]
    print('number of valid indices', len(valid_indices))


    detection_roi_scores = roi_scores[valid_indices][:max_boxes_to_draw, ...]
    detection_boxes = detection_boxes[valid_indices][:max_boxes_to_draw, ...]
    detection_masks = detection_masks[valid_indices][:max_boxes_to_draw, ...]
    detection_visual_feat = visual_features[valid_indices][:max_boxes_to_draw, ...]
    rescaled_detection_boxes = rescaled_detection_boxes[valid_indices][:max_boxes_to_draw, ...]

    #################################################################
    # Compute text embeddings and detection scores, and rank results
    text_features = build_text_embedding(categories)

image_dir = '/coco/val2017'
annotation = '/coco/annotations/instances_val2017.json'


f = open(annotation)
data = json.load(f)

images = data['images']
categories = [d['name'] for d in data['categories']]
image_path = os.path.join(image_dir, images[0]['file_name'])
inference(image_path, categories)

