# translator.py
import cv2
import numpy as np
import pyttsx3

# Import the predictor
try:
    from predict_final import predict_letter, THRESHOLD
    print("✅ Using final predictor")
except ImportError:
    try:
        from predict_enhanced import predict_letter, THRESHOLD
        print("✅ Using enhanced predictor")
    except ImportError:
        print("❌ No predictor found")
        exit(1)

# ================================
#  TEXT-TO-SPEECH SETUP
# ================================
engine = pyttsx3.init()
engine.setProperty('rate', 105)

# ================================
#  CAMERA + WINDOW SETUP
# ================================
window_name = "ASL Translator - Working!"
cv2.namedWindow(window_name)

cap = cv2.VideoCapture(0)

# ROI SIZE
roi_height, roi_width = 350, 350

# Sentence and prediction tracking
sentence = ""
prediction_history = []
current_letter = "?"
current_confidence = 0.0

print("🚀 Starting ASL Translator...")
print("📝 Place your hand in the green box and try signing A, B, C...")
print("💡 Press N to add letters, S to speak")

# ================================
#  MAIN LOOP
# ================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("No Frame Captured")
        continue

    # Fix mirrored camera (more intuitive for signing)
    frame = cv2.flip(frame, 1)

    # Center ROI
    frame_height, frame_width = frame.shape[:2]
    x_start = (frame_width - roi_width) // 2
    y_start = (frame_height - roi_height) // 2

    # Draw ROI box
    cv2.rectangle(frame, (x_start, y_start),
                  (x_start + roi_width, y_start + roi_height),
                  (0, 255, 0), 3)
    
    # Instruction text
    cv2.putText(frame, "Place hand here", 
                (x_start, y_start - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Crop ROI
    hand_roi = frame[y_start:y_start + roi_height,
                     x_start:x_start + roi_width]

    # ============================
    # Predict letter
    # ============================
    confidence, label = predict_letter(hand_roi)
    current_letter = label
    current_confidence = confidence

    # Smooth predictions (reduce flickering)
    prediction_history.append(label)
    if len(prediction_history) > 5:
        prediction_history.pop(0)
    
    # Use stable predictions
    if len(prediction_history) >= 3 and label != "?":
        stable_prediction = max(set(prediction_history), key=prediction_history.count)
        if prediction_history.count(stable_prediction) >= 3:
            current_letter = stable_prediction

    # ============================
    # DISPLAY
    # ============================
    
    # BIG prediction letter
    color = (0, 255, 0) if confidence >= THRESHOLD else (0, 0, 255)
    cv2.putText(frame, f"{current_letter}", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 5, color, 8)
    
    # Confidence bar
    bar_width = 300
    bar_height = 30
    confidence_fill = int(confidence * bar_width)
    cv2.rectangle(frame, (50, 250), (50 + bar_width, 250 + bar_height), (50, 50, 50), -1)
    cv2.rectangle(frame, (50, 250), (50 + confidence_fill, 250 + bar_height), color, -1)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (55, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ============================
    # KEY CONTROLS
    # ============================
    c = cv2.waitKey(1) & 0xFF

    if c == ord('n') or c == ord('N'):
        if current_letter != "?":
            sentence += current_letter
            print(f"✓ Added '{current_letter}' | Sentence: {sentence}")

    if c == ord('m') or c == ord('M'):
        sentence += " "
        print(f"✓ Added space | Sentence: {sentence}")

    if c == ord('d') or c == ord('D'):
        sentence = sentence[:-1] if sentence else ""
        print(f"✓ Deleted character | Sentence: {sentence}")

    if c == ord('c') or c == ord('C'):
        sentence = ""
        print("✓ Cleared sentence")

    if c == ord('s') or c == ord('S'):
        if len(sentence) > 0:
            print(f"🔊 Speaking: {sentence}")
            engine.say(sentence)
            engine.runAndWait()

    # ============================
    # SENTENCE BAR
    # ============================
    # White bar background
    cv2.rectangle(frame, (20, 20), (frame_width - 20, 100),
                  (255, 255, 255), -1)
    cv2.rectangle(frame, (20, 20), (frame_width - 20, 100),
                  (0, 0, 0), 2)

    # Sentence text
    cv2.putText(frame, f"Sentence: {sentence}", (40, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # ============================
    # HELP PANEL
    # ============================
    help_y_start = frame_height - 120
    cv2.rectangle(frame, (20, help_y_start), (frame_width - 20, frame_height - 20),
                  (0, 0, 0), -1)
    
    help_text = "N:Add  M:Space  D:Delete  C:Clear  S:Speak  ESC:Quit"
    cv2.putText(frame, help_text, (40, help_y_start + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Current prediction info
    info_text = f"Current: {current_letter} ({(current_confidence*100):.1f}%)"
    cv2.putText(frame, info_text, (40, help_y_start + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show video
    cv2.imshow(window_name, frame)

    # Exit
    if c == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print("👋 ASL Translator closed")
