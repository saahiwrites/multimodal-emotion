# Multi-Modal Emotion Detection 🎭

A comprehensive emotion detection system that analyzes emotions across multiple modalities: text, audio, and images using state-of-the-art transformer models.

## Features

- **Text Emotion Detection**: Using BERT/RoBERTa for text-based emotion analysis
- **Audio Emotion Detection**: Using Wav2Vec2 for speech emotion recognition
- **Image Emotion Detection**: Using Vision Transformer (ViT) for facial emotion recognition
- **Multi-Modal Fusion**: Combining predictions from all modalities
- **Real-time Processing**: API endpoints for live emotion detection
- **Batch Processing**: Efficient processing of large datasets

## Supported Emotions

- Anger 😠
- Disgust 🤢
- Fear 😨
- Joy 😊
- Sadness 😢
- Surprise 😲
- Neutral 😐

## Installation

```bash
git clone https://github.com/your-username/multimodal-emotion-detection.git
cd multimodal-emotion-detection
pip install -r requirements.txt
```

## Quick Start

```python
from multimodal_emotion import MultiModalEmotionDetector

detector = MultiModalEmotionDetector()

# Text emotion detection
text_emotion = detector.predict_text("I'm feeling great today!")

# Multi-modal prediction
combined_emotion = detector.predict_multimodal(
    text="I'm feeling great today!",
    audio="path/to/audio.wav",
    image="path/to/image.jpg"
)
```

## API Usage

Start the API server:
```bash
python api/app.py
```

Make predictions:
```bash
curl -X POST "http://localhost:8000/predict/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this!"}'
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
