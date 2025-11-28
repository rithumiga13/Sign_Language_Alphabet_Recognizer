# predict_final.py
import cv2
import numpy as np
import tensorflow as tf
from variables import LABELS, MODEL_PATH

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded: expects 50x50 grayscale images")

THRESHOLD = 0.7

def preprocess_hand_image(hand_roi):
    """
    Convert hand ROI to model input format: (1, 50, 50, 1)
    """
    # Convert to grayscale
    hand_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    
    # Resize to 50x50 (model's expected size)
    hand_resized = cv2.resize(hand_gray, (50, 50))
    
    # Normalize pixel values
    hand_normalized = hand_resized.astype('float32') / 255.0
    
    # Add dimensions: (50, 50) → (50, 50, 1) → (1, 50, 50, 1)
    hand_normalized = np.expand_dims(hand_normalized, axis=-1)  # Add channel
    hand_batch = np.expand_dims(hand_normalized, axis=0)        # Add batch
    
    return hand_batch

def predict_letter(hand_roi):
    """
    Predict ASL letter from hand ROI
    """
    try:
        # Preprocess image
        processed_img = preprocess_hand_image(hand_roi)
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        confidence = np.max(predictions[0])
        predicted_idx = np.argmax(predictions[0])
        predicted_letter = LABELS[predicted_idx]
        
        # Debug output (optional)
        if confidence > 0.5:
            print(f"📝 Predicted: {predicted_letter} ({confidence:.2f})")
        
        if confidence >= THRESHOLD:
            return confidence, predicted_letter
        else:
            return confidence, "?"
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return 0.0, "?"

# Backward compatibility
def which(hand_roi):
    return predict_letter(hand_roi)

print("🎯 ASL Predictor Ready!")
