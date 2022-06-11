import clip
import yaml
import numpy as np
from easydict import EasyDict
from scipy.special import softmax

from drawing import draw_output
from text_encoder import process_label
from visual_encoder import process_image

import pdb

def main(CONFIG):

    clip.available_models()
    model, _ = clip.load("ViT-B/32")
    if not CONFIG.run_on_gpu:
        model.cpu()

    (category_embedding,
        category_names,
        category_indices) = process_label(model, CONFIG)
    
    (detection_roi_scores,
        detection_boxes,
        detection_masks,
        detection_visual_feat,
        rescaled_detection_boxes,
        valid_indices,
        image_info) = process_image(CONFIG)
    #################################################################
    # Compute text embeddings and detection scores, and rank results
    raw_scores = detection_visual_feat.dot(category_embedding.T)
    if CONFIG.use_softmax:
        scores_all = softmax(CONFIG.temperature * raw_scores, axis=-1)
    else:
        scores_all = raw_scores
    #################################################################
    # Draw output and save

    draw_output(
        CONFIG,
        category_names = category_names,
        rescaled_detection_boxes = rescaled_detection_boxes,
        detection_masks = detection_masks,
        valid_indices = valid_indices,
        numbered_category_indices = category_indices,
        scores_all= scores_all,
        detection_roi_scores=detection_roi_scores,
        image_info=image_info)
    


if __name__ == "__main__":
    #pdb.set_trace()
    with open("config/test.yaml", "r") as file:
        CONFIG = yaml.load(file, Loader=yaml.SafeLoader)
        CONFIG = EasyDict(CONFIG)
    main(CONFIG)