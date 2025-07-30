# inference.py
"""
Inference utilities for emotion detection
"""
import numpy as np
from transformers import AutoTokenizer, TFLongformerForSequenceClassification

class EmotionDetector:
    """
    Emotion detection inference class
    """
    
    def __init__(self, model_path, emotions_list):
        """
        Initialize emotion detector
        
        Args:
            model_path (str): Path to saved model
            emotions_list (list): List of emotion names in order
        """
        self.emotions = emotions_list
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = TFLongformerForSequenceClassification.from_pretrained(model_path)
        print(f"✅ Emotion detector loaded from {model_path}")
    
    def predict_emotion(self, text):
        """
        Predict emotion for input text
        
        Args:
            text (str): Input text to analyze
        
        Returns:
            str: Predicted emotion name
        """
        inputs = self.tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=1024)
        logits = self.model(inputs).logits
        prediction = np.argmax(logits, axis=1)[0]
        return self.emotions[prediction]
    
    def predict_emotion_with_confidence(self, text):
        """
        Predict emotion with confidence scores
        
        Args:
            text (str): Input text to analyze
        
        Returns:
            dict: Dictionary with predicted emotion and confidence scores
        """
        inputs = self.tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=1024)
        
        logits = self.model(inputs).logits
        
        # Apply softmax to get probabilities
        import tensorflow as tf
        probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        # Create emotion-confidence mapping
        emotion_scores = {
            emotion: float(prob) for emotion, prob in zip(self.emotions, probabilities)
        }
        
        # Get predicted emotion
        predicted_emotion = self.emotions[np.argmax(probabilities)]
        
        return {
            "predicted_emotion": predicted_emotion,
            "confidence": float(probabilities.max()),
            "all_scores": emotion_scores
        }

def interactive_emotion_detection(model_path, emotions_list):
    """
    Interactive emotion detection session
    
    Args:
        model_path (str): Path to saved model
        emotions_list (list): List of emotion names
    """
    detector = EmotionDetector(model_path, emotions_list)
    
    print("\n🎭 Interactive Emotion Detection")
    print("Enter text to analyze emotions (press Enter to exit)")
    print("-" * 50)
    
    while True:
        text = input("\nEnter text: ")
        if text.strip() == "":
            print("👋 Goodbye!")
            break
        
        try:
            result = detector.predict_emotion_with_confidence(text)
            print(f"🎯 Predicted emotion: {result['predicted_emotion']}")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            
            # Show top 3 emotions
            sorted_emotions = sorted(
                result['all_scores'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            print("📈 Top 3 emotions:")
            for emotion, score in sorted_emotions:
                print(f"   {emotion}: {score:.3f}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Example usage
    model_path = "/content/drive/MyDrive/emotion_model"
    emotions_list = [
        "anger", "sadness", "joy", "disgust", "fear", 
        "surprise", "neutral"
    ]
    
    interactive_emotion_detection(model_path, emotions_list)
