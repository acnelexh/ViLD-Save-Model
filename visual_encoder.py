import numpy as np
from util import nms
from PIL import Image
import tensorflow.compat.v1 as tf

def process_image(CONFIG):
    # Load in ViLD model
    session = tf.Session(graph=tf.Graph())
    saved_model_dir = CONFIG.saved_model_dir
    _ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)
    # Get output
    roi_boxes, roi_scores, detection_boxes, scores_unused, box_outputs, detection_masks, visual_features, image_info = session.run(
        ['RoiBoxes:0', 'RoiScores:0', '2ndStageBoxes:0', '2ndStageScoresUnused:0', 'BoxOutputs:0', 'MaskOutputs:0', 'VisualFeatOutputs:0', 'ImageInfo:0'],
        feed_dict={'Placeholder:0': [CONFIG.image_path,]})
    
    roi_boxes = np.squeeze(roi_boxes, axis=0)  # squeeze
    # no need to clip the boxes, already done
    roi_scores = np.squeeze(roi_scores, axis=0)

    detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
    scores_unused = np.squeeze(scores_unused, axis=0)
    box_outputs = np.squeeze(box_outputs, axis=0)
    detection_masks = np.squeeze(detection_masks, axis=0)
    visual_features = np.squeeze(visual_features, axis=0)

    #image_info = np.squeeze(image_info, axis=0)  # obtain image info
    image_scale = np.tile(np.squeeze(image_info, axis=0)[2:3, :], (1, 2))

    rescaled_detection_boxes = detection_boxes / image_scale # rescale

    #################################################################
    # Filter boxes

    # Apply non-maximum suppression to detected boxes with nms threshold.
    nmsed_indices = nms(
        detection_boxes,
        roi_scores,
        thresh=CONFIG.nms_threshold
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
                roi_scores >= CONFIG.min_rpn_score_thresh,
                box_sizes > CONFIG.min_box_area
                )
            )    
        )
    )[0]
    print('number of valid indices', len(valid_indices))

    detection_roi_scores = roi_scores[valid_indices][:CONFIG.max_boxes_to_draw, ...]
    detection_boxes = detection_boxes[valid_indices][:CONFIG.max_boxes_to_draw, ...]
    detection_masks = detection_masks[valid_indices][:CONFIG.max_boxes_to_draw, ...]
    detection_visual_feat = visual_features[valid_indices][:CONFIG.max_boxes_to_draw, ...]
    rescaled_detection_boxes = rescaled_detection_boxes[valid_indices][:CONFIG.max_boxes_to_draw, ...]


    return detection_roi_scores, detection_boxes, detection_masks, detection_visual_feat, rescaled_detection_boxes, valid_indices, image_info

