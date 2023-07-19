import logging
import os

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils


class CutoutHandler:
    # Define the logging format
    log_format = "%(asctime)s [%(levelname)s] %(message)s"

    # Define the color codes for each log level
    color_codes = {
        "DEBUG": "\033[32m",  # Green
        "INFO": "\033[34m",  # Blue
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }

    # Define a custom logging formatter that adds color to the log messages
    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            levelname = record.levelname
            if levelname in CutoutHandler.color_codes:
                levelname_color = (
                    f"{CutoutHandler.color_codes[levelname]}{levelname}\033[0m"
                )
                record.levelname = levelname_color
            return super().format(record)

    def __init__(self, model, predictor):
        self.model = model
        self.predictor = predictor

        # Create a logger object
        self.logger = logging.getLogger(__name__)

        # Set the logging level to DEBUG
        self.logger.setLevel(logging.DEBUG)

        # Create a console handler and set its formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            CutoutHandler.ColoredFormatter(CutoutHandler.log_format)
        )

        # Add the console handler to the logger
        self.logger.addHandler(console_handler)

    def process_image(self, image: Image, name: str, prompt: str) -> list:
        self.logger.debug(f"Processing image '{name}' with prompt '{prompt}'")

        # Convert image to rgb color space
        image = image.convert("RGB")
        # Convert the image to a numpy array
        image_np = np.array(image)

        # Convert the image to RGB format
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Run the model on the image
        results = self.model(image_np)

        # Set the image for the SegmentAnything predictor
        self.predictor.set_image(image_np)

        masks_list = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                if result.names[int(box.cls[0])] == prompt:
                    self.logger.debug(
                        f"Found bounding box for prompt '{prompt}' in image '{name}'"
                    )
                    # Get the mask from the SegmentAnything predictor
                    masks, scores, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box.xyxy[0].astype(int),
                        multimask_output=False,
                    )
                    if scores[0] < 0.6:
                        self.logger.debug(
                            f"Skipping mask with score {scores[0]} for prompt '{prompt}' in image '{name}'"
                        )
                        continue
                    for i, mask in enumerate(masks):
                        self.logger.debug(
                            f"Processing mask {i+1} of {len(masks)} for prompt '{prompt}' in image '{name}'"
                        )
                        # Convert the mask numpy array to a binary mask
                        mask_binary = np.zeros(
                            (mask.shape[0], mask.shape[1]), dtype=np.uint8
                        )
                        mask_binary[mask > 0] = 1

                        # Convert the binary mask to a COCO RLE format
                        mask_rle = maskUtils.encode(np.asfortranarray(mask_binary))

                        # Extract the counts key from the mask RLE dictionary
                        counts = mask_rle["counts"]

                        # Create a new dictionary with the required keys for COCO RLE format
                        mask_coco_rle = {
                            "size": [mask.shape[0], mask.shape[1]],
                            "counts": counts.decode("utf-8"),
                        }

                        # Add the mask to the list of masks
                        masks_list.append(mask_coco_rle)

        self.logger.debug(
            f"Processed {len(masks_list)} masks for prompt '{prompt}' in image '{name}'"
        )

        # Once finished, create cutouts from the masks
        return self.create_cutout(image, name, masks_list)

    def create_cutout(self, image: Image, name: str, masks_list: list) -> list:
        cutouts = []

        for i, mask in enumerate(masks_list):
            self.logger.debug(
                f"Generating cutout {i+1} of {len(masks_list)} for image '{name}'"
            )

            # Convert the mask to a binary mask
            mask_binary = maskUtils.decode(mask).astype(np.uint8)

            # Apply the mask to the image
            image_masked = cv2.bitwise_and(image, image, mask=mask_binary)

            # Find the bounding box of the mask
            contours, _ = cv2.findContours(
                mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            x, y, w, h = cv2.boundingRect(contours[0])

            # Crop the cutout from the masked image
            cutout = image_masked[y : y + h, x : x + w]

            self.logger.debug(
                f"Cropped cutout {i+1} of {len(masks_list)} for image '{name}'"
            )

            # Convert the cutout to a PIL Image
            cutout_pil = Image.fromarray(cutout)

            self.logger.debug(
                f"Converted cutout {i+1} of {len(masks_list)} to PIL Image for image '{name}'"
            )

            # Save the cutout to a file
            cutout_filename = f"{name}_{i+1}.png"
            cutout_path = os.path.join("cutouts", cutout_filename)
            cutout_pil.save(cutout_path)

            self.logger.debug(
                f"Saved cutout {i+1} of {len(masks_list)} to file '{cutout_path}' for image '{name}'"
            )

            # Add the cutout to the list of cutouts
            cutouts.append(cutout_path)

        return cutouts
