# test_predictor.py
import cv2
import numpy as np

print("🧪 Testing predictor...")

# Test 1: Check imports
try:
    from predict_enhanced import predict_letter, THRESHOLD
    print("✅ predict_enhanced imports successfully")
except ImportError as e:
    print(f"❌ predict_enhanced import failed: {e}")

# Test 2: Check variables
try:
    from variables import LABELS, MODEL_PATH
    print(f"✅ variables imported: {len(LABELS)} labels, model path: {MODEL_PATH}")
except ImportError as e:
    print(f"❌ variables import failed: {e}")

# Test 3: Create a test image
test_image = np.ones((350, 350, 3), dtype=np.uint8) * 128  # Gray image
print("✅ Test image created")

# Test 4: Test prediction function
try:
    conf, letter = predict_letter(test_image)
    print(f"✅ Prediction test: {letter} (confidence: {conf:.2f})")
except Exception as e:
    print(f"❌ Prediction test failed: {e}")

print("🧪 Test completed")
