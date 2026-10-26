"""Facial expression recognition on FER-2013."""

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(EMOTIONS)
IMG_SIZE = 48  # native FER-2013 resolution; every model consumes 1x48x48 in [0, 1]
