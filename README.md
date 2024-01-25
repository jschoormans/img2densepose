# Img2DensePose

Adapation from the vid2densepose (https://github.com/Flode-Labs/vid2densepose/tree/main) to work with images. 

TODO
[ ] document installation of detectron2 

## Prerequisites

To utilize this tool, ensure the installation of:
- Python 3.8 or later
- PyTorch (preferably with CUDA for GPU support)
- Detectron2

## Installation Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/Flode-Labs/vid2densepose.git
    cd vid2densepose
    ```

2. Install necessary Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Clone the Detectron repository:
    ```bash
    git clone https://github.com/facebookresearch/detectron2.git
    ```

## Usage Guide

Run the script:
    
```bash
python main.py -i sample_videos/input_video.mp4 -o sample_videos/output_video.mp4
```

The script processes the input video and generates an output with the densePose format.

