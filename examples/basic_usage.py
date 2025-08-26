import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_emotion import MultiModalEmotionDetector

def main():
    detector = MultiModalEmotionDetector()

    print("=== Text Emotion Detection ===")
    text_samples = [
        "I'm so happy today!",
        "This is absolutely terrible.",
        "I'm scared of what might happen.",
        "What an amazing surprise!",
    ]

    for text in text_samples:
        result = detector.predict_text(text)
        predicted = max(result, key=result.get)
        confidence = result[predicted]
        print(f"Text: '{text}'")
        print(f"Predicted: {predicted} (confidence: {confidence:.3f})")
        print()

    print("=== Multi-Modal Emotion Detection (Text Only) ===")
    try:
        multimodal_result = detector.predict_multimodal(
            text="I'm feeling great today!"
        )
        predicted = max(multimodal_result, key=multimodal_result.get)
        print(f"Multi-modal emotion: {predicted}")
        print(f"All probabilities: {multimodal_result}")
    except Exception as e:
        print(f"Multi-modal detection error: {e}")

if __name__ == "__main__":
    main()
