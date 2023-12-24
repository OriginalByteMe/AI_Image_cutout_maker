import io
import logging
import os
from typing import Dict, List

import cv2
import numpy as np
import supervision as sv
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from modal import Image, Mount, Secret, Stub, asgi_app, method
from starlette.requests import Request
from PIL import Image

from .dino import Dino
from .s3_handler import Boto3Client
from .segment import Segmenter

# ======================
# Constants
# ======================
HOME = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
GROUNDING_DINO_CONFIG_PATH = os.path.join(
    HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
    HOME, "weights", "groundingdino_swint_ogc.pth"
)
SAM_CHECKPOINT_PATH_HIGH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
SAM_CHECKPOINT_PATH_MID = os.path.join(HOME, "weights", "sam_vit_l_0b3195.pth")
SAM_CHECKPOINT_PATH_LOW = os.path.join(HOME, "weights", "sam_vit_b_01ec64.pth")

# ======================
# Logging
# ======================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)

c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)

logger.addHandler(c_handler)


class CutoutCreator:
    def __init__(
        self,
        classes: str,
        grounding_dino_config_path: str,
        grounding_dino_checkpoint_path: str,
        encoder_version: str = "vit_b",
    ):
        self.classes = classes
        self.grounding_dino_config_path = grounding_dino_config_path
        self.grounding_dino_checkpoint_path = grounding_dino_checkpoint_path
        self.encoder_version = encoder_version
        self.HOME = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.s3 = Boto3Client()
        self.dino = Dino(
            classes=self.classes,
            box_threshold=0.35,
            text_threshold=0.25,
            model_config_path=self.grounding_dino_config_path,
            model_checkpoint_path=self.grounding_dino_checkpoint_path,
        )
        self.mask_annotator = sv.MaskAnnotator()

        encoder_checkpoint_paths = {
            "vit_b": SAM_CHECKPOINT_PATH_LOW,
            "vit_l": SAM_CHECKPOINT_PATH_MID,
            "vit_h": SAM_CHECKPOINT_PATH_HIGH,
        }

        self.sam_checkpoint_path = encoder_checkpoint_paths.get(self.encoder_version)
        self.segment = Segmenter(
            sam_encoder_version=self.encoder_version,
            sam_checkpoint_path=self.sam_checkpoint_path,
        )
    def create_annotated_image(self, image, image_name, detections):
        """Create a highlighted annotated image from an image and detections.

        Args:
            image (File): Image to be used for creating the annotated image.
            image_name (string): name of image
            detections (Detections): annotations for the image
        """
        print(f"Detections: {detections}")

        # Convert detections to masks
        detections.mask = self.segment.segment( cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections.xyxy)

        # Annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [f"{self.classes[class_id]} {confidence:0.2f}" for confidence, class_id in zip(detections.confidence, detections.class_id)]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # Convert annotated image to bytes
        img_bytes = io.BytesIO()
        Image.fromarray(np.uint8(annotated_image)).save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Upload bytes to S3
        self.s3.upload_to_s3(img_bytes.read(), "cutouts", f"{image_name}_annotated.png")

    def create_cutouts(self, image_name):
        """Create cutouts from an image and upload them to S3.

        Args:
            image_name (string): name of image
        """

        # Define paths
        data_path = os.path.join(HOME, "data")
        cutouts_path = os.path.join(HOME, "cutouts")

        # Download image from s3
        image_path = self.s3.download_from_s3(data_path, image_name)
        if image_path is None:
            logger.error(f"Failed to download image {image_name} from S3")
            return

        # Check if image exists
        if not os.path.exists(image_path):
            logger.error(f"Image {image_name} not found in folder {image_path}")
            return

        # Create cutouts directory if it doesn't exist
        os.makedirs(cutouts_path, exist_ok=True)

        # Read image
        image = cv2.imread(image_path)

        # Predict and segment image
        detections = self.dino.predict(image)
        masks = self.segment.segment(image, detections.xyxy)

        # Apply each mask to the image
        for i, mask in enumerate(masks):
            # Ensure the mask is a boolean array
            mask = mask.astype(bool)

            # Apply the mask to create a cutout
            cutout = np.zeros_like(image)
            cutout[mask] = image[mask]

            # Save the cutout
            cutout_name = f"{image_name}_cutout_{i}.png"
            cutout_path = os.path.join(cutouts_path, cutout_name)
            cv2.imwrite(cutout_path, cutout)

            # Upload the cutout to S3
            with open(cutout_path, "rb") as f:
                self.s3.upload_to_s3(f.read(), "cutouts", f"{image_name}/{cutout_name}")

        # Create annotated image
        self.create_annotated_image(image, f"{image_name}", detections)
