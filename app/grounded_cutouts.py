import os
import io
import logging
from typing import List, Dict
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from modal import asgi_app, Secret, Stub, Mount, Image, method

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
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")

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

# ======================
# FastAPI Setup
# ======================
app = FastAPI()

stub = Stub(name="cutout_generator")

origins = [
    "http://localhost:3000",  # local development
    "https://cutouts.noahrijkaard.com",  # main website
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# Modal Image Setup
# ======================
local_packages = Mount.from_local_python_packages("dino", "segment", "s3_handler")
cutout_generator_image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "segment-anything", "opencv-python", "botocore", "boto3", "fastapi", "starlette"
    )
    .run_commands(
        "apt-get update",
        "apt-get install -y git wget libgl1-mesa-glx libglib2.0-0",
        "echo $CUDA_HOME",
        "git clone https://github.com/IDEA-Research/GroundingDINO.git",
        "pip install -q -e GroundingDINO/",
        "mkdir -p /weights",
        "mkdir -p /data",
        "pip uninstall -y supervision",
        "pip uninstall -y opencv-python",
        "pip install opencv-python==4.8.0.74",
        "pip install -q supervision==0.6.0",
        "wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P weights/",
        "wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P weights/",
        "wget -q https://media.roboflow.com/notebooks/examples/dog.jpeg -P images/",
        "ls -F",
        "ls -F GroundingDINO/groundingdino/config",
        "ls -F GroundingDINO/groundingdino/models/GroundingDINO/",
    )
)

@stub.cls(
    image=cutout_generator_image,
    gpu="T4",
    mounts=[local_packages],
    secret=Secret.from_name("my-aws-secret"),
    container_idle_timeout=300,
)
class CutoutCreator:
    import cv2
    import numpy as np

    def __init__(
        self,
        classes: str,
        grounding_dino_config_path: str,
        grounding_dino_checkpoint_path: str,
        sam_checkpoint_path: str,
    ):
        self.classes = classes
        self.grounding_dino_config_path = grounding_dino_config_path
        self.grounding_dino_checkpoint_path = grounding_dino_checkpoint_path
        self.sam_checkpoint_path = sam_checkpoint_path
        self.HOME = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    def __enter__(self):
        from s3_handler import Boto3Client
        from dino import Dino
        from segment import Segmenter
        import supervision as sv

        self.dino = Dino(
            classes=self.classes,
            box_threshold=0.35,
            text_threshold=0.25,
            model_config_path=self.grounding_dino_config_path,
            model_checkpoint_path=self.grounding_dino_checkpoint_path,
        )
        self.s3 = Boto3Client()
        self.mask_annotator = sv.MaskAnnotator()
        self.segment = Segmenter(
            sam_encoder_version="vit_h", sam_checkpoint_path=self.sam_checkpoint_path
        )

    @method()
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

    @method()
    def create_cutouts(self, image_name):
        """Create cutouts from an image and upload them to S3.

        Args:
            image_name (string): name of image
        """
        import cv2
        import numpy as np

        # Define paths
        data_path = os.path.join(HOME, "data")
        cutouts_path = os.path.join(HOME, "cutouts")

        # Download image from s3
        image_path = self.s3.download_from_s3(data_path, image_name)
        if image_path is None:
            print(f"Failed to download image {image_name} from S3")
            return

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Image {image_name} not found in folder {image_path}")
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
        # self.create_annotated_image(image, f"{image_name}_{i}", detections)


@stub.local_entrypoint()
def main(
    classes: str,
    grounding_dino_config_path: str,
    grounding_dino_checkpoint_path: str,
    sam_checkpoint_path: str,
):
    return CutoutCreator(
        classes,
        grounding_dino_config_path,
        grounding_dino_checkpoint_path,
        sam_checkpoint_path,
    )

@app.get("/warmup")
async def warmup():
    """Warmup the container.

    Returns:
        _type_: return message
    """
    return "Warmed up!"

@app.post("/create-cutouts/{image_name}")
async def create_cutouts(image_name: str, request: Request):
    """
    Create cutouts from an image and upload them to S3.

    Args:
        image_name (str): Name of image to create cutouts from.
        classes (List[str], optional): A list of classes for the AI to detect for. Defaults to Body(...).

    Returns:
        _type_: _description_
    """
    try:
        # Log the start of the process
        logger.info("Creating cutouts for image %s ", image_name)

        # Parse the request body as JSON
        data = await request.json()

        # Get the classes from the JSON data
        classes = data.get("classes", [])
        logger.info("Classes: %s", classes)

        # Initialize the S3 client and the CutoutCreator
        s3 = Boto3Client()
        cutout = CutoutCreator(
            classes=classes,
            grounding_dino_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            grounding_dino_config_path=GROUNDING_DINO_CONFIG_PATH,
            sam_checkpoint_path=SAM_CHECKPOINT_PATH,
        )

        # Create the cutouts
        print(f"CREATING CUTOUTS FOR IMAGE {image_name}")
        cutout.create_cutouts.remote(image_name)
        logger.info("Cutouts created for image %s", image_name)

        # Generate presigned URLs for the cutouts
        urls = s3.generate_presigned_urls(f"cutouts/{image_name}")
        logger.info("Presigned URLs generated for cutouts of image %s", image_name)

        # Return the URLs
        return urls
    except Exception as e:
        # Log any errors that occur
        logger.error(
            "An error occurred while creating cutouts for image %s: %s", image_name, e
        )
        raise


@app.post("/create-cutouts")
async def create_all_cutouts(
    image_names: List[str] = Body(...), classes: List[str] = Body(...)
):
    """Create cutouts from multiple images and upload them to S3.

    Args:
        image_names (List[str]): List of image names to create cutouts from.
        classes (List[str], optional): A list of classes for the AI to detect for. Defaults to Body(...).

    Returns:
        Dict[str, List[str]]: A dictionary where the keys are the image names and the values are the lists of presigned URLs for the cutouts.
    """
    from s3_handler import Boto3Client

    s3 = Boto3Client()
    cutout = CutoutCreator(
        classes=classes,
        grounding_dino_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        grounding_dino_config_path=GROUNDING_DINO_CONFIG_PATH,
        sam_checkpoint_path = SAM_CHECKPOINT_PATH,
    )

    result = {}
    for image_name in image_names:
        cutout.create_cutouts(image_name)
        result[image_name] = s3.generate_presigned_urls(f"cutouts/{image_name}")

    return result


@stub.function(
    image=cutout_generator_image,
    gpu="T4",
    mounts=[local_packages],
    secret=Secret.from_name("my-aws-secret"),
    container_idle_timeout=300,
    keep_warm=1
)
@asgi_app()
def cutout_app():
    """Create a FastAPI app for creating cutouts.

    Returns:
        FastAPI: FastAPI app for creating cutouts.
    """
    return app
