import numpy as np
import torch
import argparse
import os
import cv2
import time

from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

parser = argparse.ArgumentParser()
parser.add_argument(
    '--ckpt',
    help='path to the model checkpoints',
    required=True
)
parser.add_argument(
    '--input',
    help='path to the input image',
    required=True
)
args = parser.parse_args()

out_dir = 'outputs'
os.makedirs(out_dir, exist_ok=True)

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

def image_overlay(image, segmented_image):
    alpha = 0.6 # transparency for the original image
    beta = 0.4 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum

    segmented_image = np.array(segmented_image, dtype=np.float32)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

    image = np.array(image, dtype=np.float32) / 255.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image

def load_model(ckpt):
    model_name = ckpt.split(os.path.sep)[-1]

    if 'large' in model_name:
        model_cfg = 'sam2_hiera_l.yaml'
    elif 'base_plus' in model_name:
        model_cfg = 'sam2_hiera_b+.yaml'
    elif 'small' in model_name:
        model_cfg = 'sam2_hiera_s.yaml'
    elif 'tiny' in model_name:
        model_cfg = 'sam2_hiera_t.yaml'

    model = build_sam2(
        model_cfg, ckpt, device='cuda', apply_postprocessing=False
    )
    predictor = SAM2ImagePredictor(model)

    return predictor

def get_mask(masks, random_color=False, borders=True):
    for i, mask in enumerate(masks):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]

        mask = mask.astype(np.float32)

        if i > 0:
            mask_image +=  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        else:
            mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        if borders:
            contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 

    return mask_image

show_image = cv2.imread(args.input)

# Initialize global variables
clicked = []
labels = []
rectangles = []
mode = 'point'  # Default mode
ix, iy = -1, -1
drawing = False
last_point_time = 0  # To keep track of the last point creation time
delay = 0.2  # Time delay in seconds

# Mouse callback function
def draw(event, x, y, flags, param):
    global ix, iy, drawing, rectangles, clicked, labels, mode, last_point_time

    current_time = time.time()
    
    if mode == 'point':
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append([x, y])
            labels.append(1)
            cv2.circle(show_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('image', show_image)
        elif event == cv2.EVENT_MBUTTONDOWN:
            clicked.append([x, y])
            labels.append(0)
            cv2.circle(show_image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('image', show_image)
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                if current_time - last_point_time >= delay:
                    clicked.append([x, y])
                    labels.append(1)
                    cv2.circle(show_image, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow('image', show_image)
                    last_point_time = current_time
    elif mode == 'rectangle':
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img = show_image.copy()
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('image', img)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(show_image, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', show_image)
            rectangles.append([ix, iy, x, y])

# Load an image
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)

# Press 'p' to switch to point mode, 'r' to switch to rectangle mode, 'q' to quit
while True:
    cv2.imshow('image', show_image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        mode = 'point'
        print("Switched to point mode")
    elif key == ord('r'):
        mode = 'rectangle'
        print("Switched to rectangle mode")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

input_point = np.array(clicked)
input_label = np.array(labels)
input_rectangles = np.array(rectangles)

print(input_point)
print(input_label)
print(rectangles)

image_input = np.array(Image.open(args.input).convert('RGB'))

# Load the model mask generator.
predictor = load_model(args.ckpt)
predictor.set_image(image_input)

# Inference.
masks, scores, _ = predictor.predict(
    point_coords=input_point if len(input_point) > 0 else None,
    point_labels=input_label if len(input_label) > 0 else None,
    box=rectangles if len(rectangles) > 0 else None,
    multimask_output=False,
)

rgb_mask = get_mask(
    masks, 
    borders=False
)

cv2.imshow('Image', rgb_mask)
cv2.waitKey(0)

final_image = image_overlay(image_input, rgb_mask)
cv2.imshow('Image', final_image)
cv2.waitKey(0)

cv2.imwrite(
    os.path.join(out_dir, args.input.split(os.path.sep)[-1]), 
    final_image.astype(np.float32) * 255.
)