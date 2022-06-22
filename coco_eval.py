import clip
import yaml
import numpy as np
import json
import pprint
from easydict import EasyDict
from pathlib import Path
from scipy.special import softmax
import cv2 as cv
import tensorflow.compat.v1 as tf

#from drawing import draw_output
from text_encoder import process_label
from visual_encoder import process_image
from engine import encode_visual, encode_text

import pdb

VAL_IMAGE_PATH = Path("../data/val2017")
VAL_ANNOTATION_PATH = Path("../data/annotations/instances_val2017.json")

def get_filename_id_pair(path):
    with open(path) as f:
        annotation = json.load(f)
    m_list = [(x['file_name'], x['id']) for x in annotation['images']]
    m_list.sort(key=lambda x:x[0])
    return m_list

def get_result(session, category_embedding, id_lookup, CONFIG):
    #Load param from CONFIG
    use_softmax = CONFIG.use_softmax
    temperature = CONFIG.temperature
    # Get filename and corresponding image id
    filename_id_pair = get_filename_id_pair(VAL_ANNOTATION_PATH)
    output = []
    for filename, id in filename_id_pair:
        (detection_roi_scores,
            detection_boxes,
            detection_visual_feat,
            rescaled_detection_boxes) = encode_visual(session, str(VAL_IMAGE_PATH/filename), CONFIG)
        #################################################################
        # Compute detection scores
        raw_scores = detection_visual_feat.dot(category_embedding.T)
        if use_softmax:
            scores_all = softmax(temperature * raw_scores, axis=-1)
        else:     
            scores_all = raw_scores
        pdb.set_trace()
        pred_class = np.argmax(scores_all, axis=1)
        pred_box = rescaled_detection_boxes[pred_class!=0]
        
        for idx, i in enumerate(pred_class):
            if i == 0:
                continue
            m_dict = {
                "image_id": id,
                "category_id": id_lookup[i],
                "bbox": [float(x) for x in pred_box[i]],
                "score": float(scores_all[idx][i])
            }
            output.append(m_dict)
        break
    with open("test.json", "w") as outfile:
        json_object = json.dumps(output, indent = 4)
        outfile.write(json_object)


def main(CONFIG):
    # Load in CLIP and generate text embedding for COCO
    clip.available_models()
    model, _ = clip.load("ViT-B/32")
    if not CONFIG.run_on_gpu:
        model.cpu()
    category_embedding, id_lookup = encode_text(model, VAL_ANNOTATION_PATH, CONFIG)
    #print(id_lookup)
    #print(category_embedding.shape)
    # Load in ViLD model
    session = tf.Session(graph=tf.Graph())
    saved_model_dir = CONFIG.saved_model_dir
    _ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)

    # Get all the filename and id pair
    get_result(session, category_embedding, id_lookup)

if __name__ == "__main__":
    #pdb.set_trace()
    with open("config/eval.yaml", "r") as file:
        CONFIG = yaml.load(file, Loader=yaml.SafeLoader)
        CONFIG = EasyDict(CONFIG)
    main(CONFIG)