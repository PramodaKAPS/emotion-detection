# main.py
"""

"""
import os
from setup import setup_cache_directory
from train import train_emotion_model

def main():
   
    print("🚀 Starting Emotion Detection Training")
    print("=" * 60)
    
    # Setup cache directory
    cache_dir = setup_cache_directory()
    
    # Training configuration
    save_path = "./emotion_model" 
    os.makedirs(save_path, exist_ok=True) 
    selected_emotions = [
        "anger", "sadness", "joy", "disgust", "fear", 
        "surprise", "neutral"
    ]

    
    config = {
        "num_train": 5000,
        "num_epochs": 6,
        "batch_size": 8
    }
    
    print(f" Training Configuration:")
    print(f"   - Cache directory: {cache_dir}")
    print(f"   - Save path: {save_path}")
    print(f"   - Selected emotions: {selected_emotions}")
    print(f"   - Training samples: {config['num_train']}")
    print(f"   - Epochs: {config['num_epochs']}")
    print(f"   - Batch size: {config['batch_size']}")
    print("-" * 60)
    
    # Run training
    try:
        test_results = train_emotion_model(
            cache_dir=cache_dir,
            save_path=save_path,
            selected_emotions=selected_emotions,
            num_train=config["num_train"],
            epochs=config["num_epochs"],
            batch_size=config["batch_size"]
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"📊 Final test results: {test_results}")
        print(f"💾 Model saved to: {save_path}")
        
    except Exception as e:
        print(f" Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
