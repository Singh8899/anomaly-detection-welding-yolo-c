import json
from dataclasses import dataclass
from typing import Any, Union


@dataclass
class Detection:
    bbox: list[float]
    class_name: str
    score: float

    def _str_(self) -> str:
        return f"{self.class_name} - {self.score:.2f} - bbox: {self.bbox}"

    def _repr_(self) -> str:
        return self._str_()

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> 'Detection':
        return cls(
            bbox=json_data.get("bbox", []),
            class_name=json_data.get("class_name", "unknown"),
            score=json_data.get("score", 0.0)
        )


@dataclass
class InferenceResponse:
    predictions: list[Detection]

    def _get_dumped_predictions(self) -> list[dict[str, Union[list[float], str, float]]]:
        return [
            {
                "bbox": detection.bbox,
                "class_name": detection.class_name,
                "score": detection.score
            } for detection in self.predictions
        ]

    def to_json(self) -> dict[str, list[dict[str, Union[list[float], str, float]]]]:
        return json.dumps({
            "predictions": self._get_dumped_predictions()
        })

    def is_empty(self) -> bool:
        return len(self.predictions) == 0

    def has_errors(self) -> bool:
        return any(detection.class_name.lower() == "bad" for detection in self.predictions)

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> 'InferenceResponse':
        return cls(predictions=[Detection(**detection) for detection in json_data.get("predictions", [])])