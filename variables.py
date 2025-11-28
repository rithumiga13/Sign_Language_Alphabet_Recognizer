"""
Contains variables for image-size, model path and class labels.
"""

import os

# ✅ Image size used during training
IMAGE_SIZE = 50       # Your model was trained on 50x50 grayscale images
NUM_OF_CHANNELS = 1   # Grayscale

# ✅ Path to saved model (choose one)
MODEL_PATH = "model.keras"   # OR "model.h5"

# ✅ Path to dataset (required for LABELS loading)
DATA_PATH = "data/asl"       # Adjust if your folder is different

# ✅ Label list A–Z
LABELS = sorted(os.listdir(DATA_PATH))

# ✅ Minimum confidence threshold
THRESHOLD = 25

