import io
import os
from typing import Dict

import cv2
import numpy as np
import supervision as sv
from dino import Dino
from PIL import Image
from s3_handler import Boto3Client
from segment import Segmenter

HOME = os.path.abspath(os.path.join(os.getcwd(), os.pardir))


class CutoutCreator:
    """
    A class for creating cutouts from an image and uploading them to S3.

    Attributes:
      dino: A Dino object for object detection.
      s3: A Boto3Client object for uploading to S3.
      mask_annotator: A MaskAnnotator object for annotating images with masks.
    """

    def __init__(
        self,
        classes: str,
        grounding_dino_config_path: str,
        grounding_dino_checkpoint_path: str,
    ):
        self.dino = Dino(
            classes=classes,
            box_threshold=0.35,
            text_threshold=0.25,
            model_config_path=grounding_dino_config_path,
            model_checkpoint_path=grounding_dino_checkpoint_path,
        )
        self.s3 = Boto3Client()
        self.mask_annotator = sv.MaskAnnotator()

    def create_annotated_image(self, image, image_name, detections: Dict[str, list]):
        """Create a highlighted annotated image from an image and detections.

        Args:
            image (File): Image to be used for creating the annotated image.
            image_name (string): name of image
            detections (Dict[str, list]): annotations for the image
        """
        annotated_image = self.mask_annotator.annotate(
            scene=image, detections=detections
        )
        # Convert annotated image to bytes
        img_bytes = io.BytesIO()
        Image.fromarray(np.uint8(annotated_image)).save(img_bytes, format="PNG")
        img_bytes.seek(0)
        # Upload bytes to S3
        self.s3.upload_to_s3(img_bytes.read(), "cutouts", f"{image_name}_annotated.png")

    def create_cutouts(self, image_name, sam_checkpoint_path):
        """Create cutouts from an image and upload them to S3.

        Args:
            image_name (string): name of image
            sam_checkpoint_path (string): path to sam checkpoint
        """
        # Download image from s3
        image_path = self.s3.download_from_s3(os.path.join(HOME, "data"), image_name)
        if not os.path.exists(os.path.join(HOME, "cutouts")):
            os.mkdir(os.path.join(HOME, "cutouts"))
        image = cv2.imread(image_path)
        segment = Segmenter(
            sam_encoder_version="vit_h", sam_checkpoint_path=sam_checkpoint_path
        )
        detections = self.dino.predict(image)

        masks = segment.segment(image, detections.xyxy)
        # Load the image
        # image_path = os.path.join(self.image_folder, image_name)
        # for item in os.listdir(self.image_folder):
        #   print("Item: ",item)
        if not os.path.exists(image_path):
            print(f"Image {image_name} not found in folder {image_path}")
            return

        image = cv2.imread(image_path)

        # Apply each mask to the image
        for i, mask in enumerate(masks):
            # Ensure the mask is a boolean array
            mask = mask.astype(bool)

            # Apply the mask to create a cutout
            cutout = np.zeros_like(image)
            cutout[mask] = image[mask]

            # Save the cutout
            cutout_name = f"{image_name}_cutout_{i}.png"
            cutout_path = os.path.join(HOME, "cutouts", cutout_name)
            cv2.imwrite(cutout_path, cutout)

            # Upload the cutout to S3
            with open(cutout_path, "rb") as f:
                self.s3.upload_to_s3(f.read(), "cutouts", f"{image_name}/{cutout_name}")

        # Create annotated image
        # self.create_annotated_image(image, f"{image_name}_{i}", detections)
