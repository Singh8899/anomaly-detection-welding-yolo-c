import logging

import cv2
import numpy as np
from classes import InferenceResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def draw_boxes(orig_image: np.ndarray, inference_response: InferenceResponse, inference_size: tuple) -> np.ndarray:
    print
    logger.info(f"Drawing boxes on image with shape: {orig_image.shape}")
    orig_h, orig_w, _ = orig_image.shape

    # Calculate dimensions to maintain aspect ratio
    aspect_ratio = orig_w / orig_h
    inf_h = int(inference_size / aspect_ratio)

    logger.info(
        f"Inference dimensions: {inference_size}x{inf_h}, original dimensions: {orig_w}x{orig_h}")

    scale_x = 1
    scale_y = 1

    lw = max(round((orig_w + orig_h) / 2 * 0.003), 2)
    font_scale = lw / 3.8
    font_thickness = max(lw - 1, 1)

    annotated_image = orig_image.copy()

    logger.info(f"Drawing boxes on image with shape: {orig_image.shape}")
    for detection in inference_response.predictions:
        logger.info(f"{detection}")
        # Expected format: [x1, y1, x2, y2] on the resized image.
        bbox = detection.bbox

        x1 = int(bbox[0] * scale_x)
        y1 = int(bbox[1] * scale_y)
        x2 = int(bbox[2] * scale_x)
        y2 = int(bbox[3] * scale_y)

        class_lower = detection.class_name.lower()
        print(f"Drawing box for class: {class_lower} with score: {detection.score}")
        if class_lower == "class_1":
            color = (0, 255, 0)  # Green
        elif class_lower == "class_0":
            color = (0, 0, 255)  # Red
        else:
            color = (255, 0, 0)

        cv2.rectangle(annotated_image, (x1, y1),
                      (x2, y2), color=color, thickness=lw)

        label = f"{detection.score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        cv2.rectangle(
            annotated_image,
            (x1, y1 - text_height - baseline),
            (x1 + text_width, y1),
            color=color,
            thickness=-1
        )

        cv2.putText(
            annotated_image,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA
        )

    return annotated_image