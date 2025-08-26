import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from typing import Dict, List

class AudioEmotionDetector:
    def __init__(self, config: Dict):
        self.model_name = config.get("model_name", "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        self.sample_rate = config.get("sample_rate", 16000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )

        return inputs.input_values.to(self.device)

    def predict(self, audio_path: str) -> Dict[str, float]:
        inputs = self.preprocess_audio(audio_path)

        with torch.no_grad():
            outputs = self.model(inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            probabilities = probabilities.cpu().numpy()[0]

        return dict(zip(self.emotions, probabilities.tolist()))

    def predict_batch(self, audio_paths: List[str]) -> List[Dict[str, float]]:
        results = []
        for audio_path in audio_paths:
            result = self.predict(audio_path)
            results.append(result)
        return results
