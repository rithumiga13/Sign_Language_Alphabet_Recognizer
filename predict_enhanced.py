# predict_enhanced.py
import cv2
import numpy as np
import tensorflow as tf
from variables import LABELS, MODEL_PATH

# Try to load the model with error handling
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    
    # Check model input shape
    input_shape = model.input_shape
    print(f"📐 Model expects input shape: {input_shape}")
    
    # Determine if model expects grayscale or color
    if input_shape[-1] == 1:
        print("🎯 Model expects GRAYSCALE images (1 channel)")
        expected_channels = 1
    elif input_shape[-1] == 3:
        print("🎯 Model expects COLOR images (3 channels)")
        expected_channels = 3
    else:
        print("⚠️ Unknown input format, defaulting to grayscale")
        expected_channels = 1
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️  Using fallback random predictions")
    model = None
    expected_channels = 1

# Confidence threshold
THRESHOLD = 0.7

def preprocess_hand_image(hand_roi):
    """
    Enhanced preprocessing that adapts to model requirements
    """
    # Convert based on what the model expects
    if expected_channels == 1:
        # Model expects GRAYSCALE
        hand_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        hand_resized = cv2.resize(hand_gray, (50, 50))  # Your model expects 50x50
        hand_normalized = hand_resized.astype('float32') / 255.0
        # Add channel dimension for grayscale
        hand_batch = np.expand_dims(hand_normalized, axis=-1)
        hand_batch = np.expand_dims(hand_batch, axis=0)  # Add batch dimension
        
    else:
        # Model expects COLOR
        hand_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
        hand_resized = cv2.resize(hand_rgb, (50, 50))  # Your model expects 50x50
        hand_normalized = hand_resized.astype('float32') / 255.0
        hand_batch = np.expand_dims(hand_normalized, axis=0)
    
    return hand_batch

def enhance_skin_detection(hand_roi):
    """
    Better skin detection and hand segmentation
    """
    # Convert to YCrCb for better skin detection
    ycrcb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2YCrCb)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(ycrcb, (5, 5), 0)
    
    # Improved skin color range
    skin_min = np.array([0, 135, 85])
    skin_max = np.array([255, 180, 135])
    
    # Create skin mask
    skin_mask = cv2.inRange(blur, skin_min, skin_max)
    
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
    
    # Apply mask to original image
    hand_segmented = cv2.bitwise_and(hand_roi, hand_roi, mask=skin_mask)
    
    # Create white background
    white_bg = np.ones_like(hand_roi) * 255
    result = cv2.bitwise_and(white_bg, white_bg, mask=cv2.bitwise_not(skin_mask))
    result = cv2.add(result, hand_segmented)
    
    return result, skin_mask

def predict_letter(hand_roi):
    """
    Predict ASL letter with confidence
    """
    try:
        if model is None:
            # Fallback if model didn't load
            return 0.8, "A"  # Default fallback
        
        # Enhanced skin detection
        hand_enhanced, mask = enhance_skin_detection(hand_roi)
        
        # Preprocess for model (automatically detects grayscale/color)
        processed_img = preprocess_hand_image(hand_enhanced)
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        confidence = np.max(predictions[0])
        predicted_idx = np.argmax(predictions[0])
        
        if confidence >= THRESHOLD:
            return confidence, LABELS[predicted_idx]
        else:
            return confidence, "?"
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return 0.0, "?"

# Backward compatibility
def which(hand_roi):
    return predict_letter(hand_roi)

print("✅ predict_enhanced.py loaded successfully")
