import yaml
import numpy as np
from typing import Dict, List
import os

def load_config(config_path: str) -> Dict:
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        else:
            return get_default_config()
    except Exception:
        return get_default_config()

def get_default_config() -> Dict:
    return {
        "text": {
            "model_name": "j-hartmann/emotion-english-distilroberta-base",
            "max_length": 512
        },
        "audio": {
            "model_name": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            "sample_rate": 16000
        },
        "image": {
            "model_name": "trpakov/vit-face-expression",
            "image_size": 224
        },
        "fusion_weights": {
            "text": 0.4,
            "audio": 0.3,
            "image": 0.3
        }
    }

def combine_predictions(
    predictions: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
    emotions: List[str]
) -> Dict[str, float]:
    combined_scores = {emotion: 0.0 for emotion in emotions}
    total_weight = 0.0

    for modality, pred in predictions.items():
        if modality in weights:
            weight = weights[modality]
            total_weight += weight

            for emotion in emotions:
                if emotion in pred:
                    combined_scores[emotion] += pred[emotion] * weight

    if total_weight > 0:
        combined_scores = {
            emotion: score / total_weight 
            for emotion, score in combined_scores.items()
        }

    return combined_scores

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
