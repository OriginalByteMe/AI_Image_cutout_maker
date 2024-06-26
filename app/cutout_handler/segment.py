from app.common import cutout_handler_app, cutout_generator_image


with cutout_generator_image.imports():
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    import numpy as np


cutout_handler_app.cls()


class Segmenter:
    import numpy as np

    def __init__(
        self,
        sam_encoder_version: str,
        sam_checkpoint_path: str,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = sam_model_registry[sam_encoder_version](
            checkpoint=sam_checkpoint_path
        ).to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)

    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box, multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
