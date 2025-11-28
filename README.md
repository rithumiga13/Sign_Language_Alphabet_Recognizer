

# American Sign Language (ASL) Alphabet Recognition  
Real-time ASL alphabet translator using Python, OpenCV, TensorFlow-Metal, and a CNN model.

This project implements a real-time static hand-sign classifier that recognizes ASL alphabets from webcam input. It includes a custom-trained CNN model, a prediction module, and an interactive translator interface with text-to-speech.



Installation
1. Clone the repository
```bash
git clone https://github.com/<your-username>/Sign_language_alphabet_recognizer.git
cd Sign_language_alphabet_recognizer
````

### 2. Create and activate a virtual environment

```bash
python3 -m venv asl_env_310
source asl_env_310/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For Apple Silicon (M1/M2/M3), install:

```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

---

## Dataset

The project uses the ASL Alphabet dataset containing:

* 26 classes (A to Z)
* Approximately 700 images per class
* About 18,200 total images
* RGB images resized to 64×64×3 for training

Dataset link:
[https://www.kaggle.com/datasets/ayuraj/asl-dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset)

---

## Training the Model

Training is performed in the notebook:

```
ASL.ipynb
```

The final trained model is saved as:

```
model.keras
model.h5
```

Update `variables.py` accordingly:

```python
MODEL_PATH = "model.keras"
THRESHOLD = 75
IMAGE_SIZE = 64
LABELS = ['A','B','C', ..., 'Z']
```

---

## Prediction Pipeline

`predict.py` handles:

* Image preprocessing (resize → RGB → normalization)
* Preparing image for CNN input
* Running the model to get class probabilities
* Returning predicted label and confidence

Usage example:

```bash
from predict import which
confidence, letter = which(image)
```

---

## Real-Time Translator

Run the translator:

```bash
python translator.py
```

Features:

* Live webcam feed
* Mirrored camera view
* Large Region of Interest (ROI)
* Skin-mask-based background removal
* Display of predicted letter and confidence
* Sentence assembly and display bar
* Integrated offline text-to-speech (pyttsx3)

Keyboard controls:

| Key | Function              |
| --- | --------------------- |
| N   | Add predicted letter  |
| M   | Add space             |
| D   | Delete last character |
| C   | Clear sentence        |
| S   | Speak sentence        |
| ESC | Quit                  |

---

## Requirements

```
opencv-python
numpy
tensorflow-macos
tensorflow-metal
keras
pyttsx3
matplotlib
```


---

## Future Improvements

* Dynamic gesture recognition
* Model upgrade to MobileNet/EfficientNet
* Improved preprocessing and skin detection
* Gesture smoothing and temporal filtering
* Background subtraction for higher accuracy

---

## Contributing

Contributions and suggestions are welcome. Open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License.

```

---

If you want a **shorter version**, **academic-style version**, or **resume-optimized version**, I can generate it.
```
