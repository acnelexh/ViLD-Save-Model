
import numpy as np
from PIL import Image
from matplotlib import patches
from matplotlib import pyplot as plt
from util import paste_instance_masks, visualize_boxes_and_labels_on_image_array, display_image, plot_mask

# Global matplotlib settings
SMALL_SIZE = 16#10
MEDIUM_SIZE = 18#12
BIGGER_SIZE = 20#14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Parameters for drawing figure.
display_input_size = (10, 10)
overall_fig_size = (18, 24)

line_thickness = 2
fig_size_w = 35
mask_color =   'red'
alpha = 0.5

def draw_output(CONFIG, **kwargs):
    # Unpack args
    category_names = kwargs["category_names"]
    rescaled_detection_boxes = kwargs["rescaled_detection_boxes"]
    detection_masks = kwargs["detection_masks"]
    valid_indices = kwargs["valid_indices"]
    numbered_category_indices= kwargs["numbered_category_indices"]
    scores_all= kwargs["scores_all"]
    detection_roi_scores = kwargs["detection_roi_scores"]
    image_info = kwargs["image_info"]

    # Unpack CONFIG
    image_path = CONFIG.image_path
    max_boxes_to_draw = CONFIG.max_boxes_to_draw
    min_rpn_score_thresh = CONFIG.min_rpn_score_thresh

    #################################################################
    # Rank result based on score
    indices = np.argsort(-np.max(scores_all, axis=1))  # Results are ranked by scores
    indices_fg = np.array([i for i in indices if np.argmax(scores_all[i]) != 0])
    fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)

    image_info = np.squeeze(image_info, axis=0)  # obtain image info
    image_scale = np.tile(image_info[2:3, :], (1, 2))
    image_height = int(image_info[0, 0])
    image_width = int(image_info[0, 1])

    image = np.asarray(Image.open(open(image_path, 'rb')).convert("RGB"))
    #################################################################
    # Plot detected boxes on the input image.
    ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)
    processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
    segmentations = paste_instance_masks(detection_masks, processed_boxes, image_height, image_width)

    if len(indices_fg) == 0:
        display_image(np.array(image), image_path, size=overall_fig_size)
        print('ViLD does not detect anything belong to the given category')

    else:
        image_with_detections = visualize_boxes_and_labels_on_image_array(
            np.array(image),
            rescaled_detection_boxes[indices_fg],
            valid_indices[:max_boxes_to_draw][indices_fg],
            detection_roi_scores[indices_fg],    
            numbered_category_indices,
            instance_masks=segmentations[indices_fg],
            use_normalized_coordinates=False,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_rpn_score_thresh,
            skip_scores=False,
            skip_labels=True)

        plt.figure(figsize=overall_fig_size)
        plt.imshow(image_with_detections)
        plt.axis('off')
        plt.title('Detected objects and RPN scores')
        plt.show()
        plt.savefig(f'{CONFIG.save_path}/result.png')

    #################################################################
    # Plot
    cnt = 0
    raw_image = np.array(image)
    n_boxes = rescaled_detection_boxes.shape[0]

    for anno_idx in indices[0:int(n_boxes)]:
        rpn_score = detection_roi_scores[anno_idx]
        bbox = rescaled_detection_boxes[anno_idx]
        scores = scores_all[anno_idx]
        if np.argmax(scores) == 0:
            continue
        
        y1, x1, y2, x2 = int(np.floor(bbox[0])), int(np.floor(bbox[1])), int(np.ceil(bbox[2])), int(np.ceil(bbox[3]))
        img_w_mask = plot_mask(mask_color, alpha, raw_image, segmentations[anno_idx])
        crop_w_mask = img_w_mask[y1:y2, x1:x2, :]


        fig, axs = plt.subplots(1, 4, figsize=(fig_size_w, fig_size_h), gridspec_kw={'width_ratios': [3, 1, 1, 2]}, constrained_layout=True)

        # Draw bounding box.
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=line_thickness, edgecolor='r', facecolor='none')
        axs[0].add_patch(rect)

        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title(f'bbox: {y1, x1, y2, x2} area: {(y2 - y1) * (x2 - x1)} rpn score: {rpn_score:.4f}')
        axs[0].imshow(raw_image)

        # Draw image in a cropped region.
        crop = np.copy(raw_image[y1:y2, x1:x2, :])
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        
        axs[1].set_title(f'predicted: {category_names[np.argmax(scores)]}')
        axs[1].imshow(crop)

        # Draw segmentation inside a cropped region.
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[2].set_title('mask')
        axs[2].imshow(crop_w_mask)

        # Draw category scores.
        fontsize = max(min(fig_size_h / float(len(category_names)) * 45, 20), 8)
        for cat_idx in range(len(category_names)):
            axs[3].barh(cat_idx, scores[cat_idx], 
                        color='orange' if scores[cat_idx] == max(scores) else 'blue')
            axs[3].invert_yaxis()
            axs[3].set_axisbelow(True)
            axs[3].set_xlim(0, 1)
            plt.xlabel("confidence score")
            axs[3].set_yticks(range(len(category_names)))
            axs[3].set_yticklabels(category_names, fontdict={
                'fontsize': fontsize})
            
            cnt += 1
            # fig.tight_layout()
        plt.savefig(f"{CONFIG.save_path}/result_{anno_idx}.png")
    print('Detection counts:', cnt)
