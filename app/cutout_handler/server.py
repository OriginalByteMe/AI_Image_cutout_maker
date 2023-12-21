import logging
import os
from typing import Dict, List

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from modal import Secret, asgi_app
from starlette.requests import Request

from app.common import cutout_handler_stub, local_packages

from .grounded_cutouts import CutoutCreator

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


# ======================
# FastAPI Setup
# ======================
app = FastAPI()

# stub = Stub(name="cutout_generator")

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


@app.get("/warmup")
async def warmup():
    """Warmup the container.

    Returns:
        _type_: return message
    """
    # Spins up the container and loads the models, this will make it easier to create cutouts later
    CutoutCreator(
        classes=[],
        grounding_dino_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        grounding_dino_config_path=GROUNDING_DINO_CONFIG_PATH,
        encoder_version="vit_b",
    )

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
    from s3_handler import Boto3Client

    try:
        # Log the start of the process
        logger.info("Creating cutouts for image %s ", image_name)

        # Parse the request body as JSON
        data = await request.json()

        # Get the classes and accuracy level from the JSON data
        classes = data.get("classes", [])
        accuracy_level = data.get("accuracy_level", "low")
        logger.info("Classes: %s", classes)
        logger.info("Accuracy level: %s", accuracy_level)

        # Select the SAM checkpoint path based on the accuracy level
        accuracy_encoder_versions = {
            "high": "vit_h",
            "mid": "vit_l",
            "low": "vit_b",
        }
        encoder_version = accuracy_encoder_versions.get(accuracy_level, "vit_b")

        # Initialize the S3 client and the CutoutCreator
        s3 = Boto3Client()
        cutout = CutoutCreator(
            classes=classes,
            grounding_dino_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            grounding_dino_config_path=GROUNDING_DINO_CONFIG_PATH,
            encoder_version=encoder_version,
        )

        # Create the cutouts
        print(f"CREATING CUTOUTS FOR IMAGE {image_name}")
        cutout.create_cutouts(image_name)
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
        encoder_version="vit_b",
    )

    result = {}
    for image_name in image_names:
        cutout.create_cutouts(image_name)
        result[image_name] = s3.generate_presigned_urls(f"cutouts/{image_name}")

    return result


@cutout_handler_stub.function(
    gpu="T4",
    mounts=[local_packages],
    secret=Secret.from_name("my-aws-secret"),
    container_idle_timeout=300,
    retries=1,
)
@asgi_app()
def cutout_app():
    """Create a FastAPI app for creating cutouts.

    Returns:
        FastAPI: FastAPI app for creating cutouts.
    """
    return app
