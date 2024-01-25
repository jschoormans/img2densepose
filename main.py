import argparse
import os
import PIL
import cv2
import numpy as np
import torch
from densepose import add_densepose_config
from densepose.vis.densepose_results import (
    DensePoseResultsFineSegmentationVisualizer as Visualizer,
)
from densepose.vis.extractor import DensePoseResultExtractor

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import time
from tqdm import tqdm
def get_image_files(folder_path):
    image_files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and is_image_file(file_path):
            image_files.append(file_path)
    return image_files

def is_image_file(file_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)

def main(input_path, output_path):
    
    predictor = create_predictor()
    if os.path.isfile(input_path):
        # Process a single image
        process_image(input_path, output_path, predictor)
    elif os.path.isdir(input_path):
        # Process all images in the folder
        image_files = get_image_files(input_path)
        total_images = len(image_files)
        processed_images = 0
        start_time = time.time()

        for image_file in tqdm(image_files):
            image_name = os.path.splitext(os.path.basename(image_file))[0]
            output_file = os.path.join(output_path, f"{image_name}_densepose.jpg")

            if os.path.isfile(output_file):
                print(f"Output file {output_file} already exists. Skipping...")
                continue

            process_image(image_file, output_file, predictor)
            processed_images += 1

            # Calculate progress and time estimate
            progress = processed_images / total_images * 100
            elapsed_time = time.time() - start_time
            estimated_time = elapsed_time / processed_images * (total_images - processed_images)


    else:
        print("Invalid input path.")

def create_predictor():
    # Initialize Detectron2 configuration for DensePose
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file("detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    return predictor    


def process_image(input_image_path, output_image_path, predictor):

    # Open the input image
    frame = cv2.imread(input_image_path)

    with torch.no_grad():
        outputs = predictor(frame)["instances"]

    results = DensePoseResultExtractor()(outputs)

    # MagicAnimate uses the Viridis colormap for their training data
    cmap = cv2.COLORMAP_VIRIDIS
    # Visualizer outputs black for background, but we want the 0 value of
    # the colormap, so we initialize the array with that value
    arr = cv2.applyColorMap(np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8), cmap)
    out_frame = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)

    # Save the output image
    cv2.imwrite(output_image_path, out_frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", type=str, default="./input_image.jpg"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="./output"
    )
    args, _ = parser.parse_known_args()  # Use parse_known_args to ignore unrecognized arguments

    main(args.input_path, args.output_path)
