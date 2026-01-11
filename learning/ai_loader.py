"""Lazy loader for ASL and Word models.

This module exposes `get_asl_recognizer` and `get_word_recognizer` which
initialize the recognizers on first use and return the global instance.
It avoids importing/initializing heavy ML objects at Django settings import
time.
"""
import os
from threading import Lock

_asl_initialized = False
_word_initialized = False
_init_lock = Lock()

def get_asl_recognizer(model_path=None):
    """Ensure ASL recognizer is initialized and return the instance.

    Returns the recognizer object or None if initialization failed.
    """
    from .ai_recognizer import get_recognizer, init_recognizer

    recognizer = get_recognizer()
    if recognizer is not None:
        return recognizer

    # lazy init
    with _init_lock:
        recognizer = get_recognizer()
        if recognizer is not None:
            return recognizer
        if model_path and os.path.exists(model_path):
            try:
                ok = init_recognizer(model_path)
                if ok:
                    return get_recognizer()
            except Exception:
                return None
    return None

def get_word_recognizer_lazy(model_path=None):
    """Ensure Word recognizer is initialized and return the instance.

    Returns the recognizer object or None if initialization failed.
    """
    from .word_recognizer import get_word_recognizer, init_word_recognizer

    recognizer = get_word_recognizer()
    if recognizer is not None:
        return recognizer

    with _init_lock:
        recognizer = get_word_recognizer()
        if recognizer is not None:
            return recognizer
        if model_path and os.path.exists(model_path):
            try:
                ok = init_word_recognizer(model_path)
                if ok:
                    return get_word_recognizer()
            except Exception:
                return None
    return None

def get_digit_recognizer(model_path=None):
    """Ensure Digit recognizer is initialized and return the instance.

    Returns the recognizer object or None if initialization failed.
    """
    from .digit_recognizer import get_digit_recognizer_instance, init_digit_recognizer

    recognizer = get_digit_recognizer_instance()
    if recognizer is not None:
        return recognizer

    with _init_lock:
        recognizer = get_digit_recognizer_instance()
        if recognizer is not None:
            return recognizer
        if model_path and os.path.exists(model_path):
            try:
                ok = init_digit_recognizer(model_path)
                if ok:
                    return get_digit_recognizer_instance()
            except Exception:
                return None
        else:
            # Try to find model file
            possible_paths = [
                "yolov8_asl_digits_final.h5",
                "../yolov8_asl_digits_final.h5",
                "./yolov8_asl_digits_final.h5",
                "models/yolov8_asl_digits_final.h5",
                "../models/yolov8_asl_digits_final.h5",
                os.path.join(os.path.dirname(__file__), "yolov8_asl_digits_final.h5"),
                os.path.join(os.path.dirname(__file__), "..", "yolov8_asl_digits_final.h5"),
                os.path.join(os.path.dirname(__file__), "..", "models", "yolov8_asl_digits_final.h5")
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        ok = init_digit_recognizer(path)
                        if ok:
                            return get_digit_recognizer_instance()
                    except Exception:
                        continue
    return None