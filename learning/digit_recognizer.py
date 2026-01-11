# learning/digit_recognizer.py
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import traceback
import mediapipe as mp
from collections import deque

class DigitRecognizer:
    """Nhận diện chữ số ASL (0-9) sử dụng MobileNetV2"""
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.img_size = 224
        self.class_names = [str(i) for i in range(10)]  # 0-9
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
            static_image_mode=False,
            model_complexity=1
        )
        
        # Smoothing buffer and detection state
        self.prediction_buffer = deque(maxlen=5)
        self.last_hand_detected = False
        self.last_bbox = None
        
    def initialize(self, model_path):
        """Khởi tạo model nhận diện chữ số"""
        try:
            print(f"🚀 Initializing ASL Digit Recognizer from: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                return False
            
            # Load model
            self.model = keras.models.load_model(model_path)
            print("✅ ASL Digit Model loaded successfully!")
            
            # Test model với input giả
            test_input = np.random.randn(1, self.img_size, self.img_size, 3).astype(np.float32)
            prediction = self.model.predict(test_input, verbose=0)
            print(f"🧪 Model test - Output shape: {prediction.shape}")
            
            self.is_initialized = True
            print("✅ ASL Digit Recognizer initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing digit recognizer: {e}")
            traceback.print_exc()
            return False
    
    def process_frame(self, frame):
        """Process frame with MediaPipe ROI crop + smoothing (like desktop script)"""
        try:
            # Convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.hands.process(rgb)
            
            self.last_hand_detected = False
            
            if not results.multi_hand_landmarks:
                self.prediction_buffer.clear()
                self.last_bbox = None
                return "--", 0.0, None
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand bounding box
                h, w, _ = frame.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min = int(min(x_coords) * w) - 30
                x_max = int(max(x_coords) * w) + 30
                y_min = int(min(y_coords) * h) - 30
                y_max = int(max(y_coords) * h) + 30
                
                # Ensure within frame bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)
                
                # Skip if ROI too small
                if x_max - x_min < 10 or y_max - y_min < 10:
                    continue
                
                # Extract ROI
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    continue
                
                # Preprocess ROI
                img = cv2.resize(roi, (self.img_size, self.img_size))
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)
                
                # Predict
                preds = self.model.predict(img, verbose=0)
                class_id = np.argmax(preds)
                confidence = float(np.max(preds))
                prediction = self.class_names[class_id]
                
                self.last_hand_detected = True
                self.last_bbox = (x_min, y_min, x_max, y_max)
                self.prediction_buffer.append((prediction, confidence))
                
                smooth_pred, smooth_conf = self._get_smoothed_prediction()
                return smooth_pred, smooth_conf, self.last_bbox
            
            # No valid hand found
            self.last_hand_detected = False
            self.last_bbox = None
            return "--", 0.0, None
            
        except Exception as e:
            print(f"❌ Error in digit process_frame: {e}")
            return "--", 0.0, None
    
    def _get_smoothed_prediction(self):
        """Smooth predictions using buffer"""
        if not self.prediction_buffer:
            return "--", 0.0
        
        labels = [label for label, _ in self.prediction_buffer]
        dominant = max(set(labels), key=labels.count)
        confidences = [conf for label, conf in self.prediction_buffer if label == dominant]
        
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return dominant, float(avg_conf)
    
    def preprocess_frame(self, frame):
        """Tiền xử lý frame cho model (legacy full-frame)"""
        try:
            # Resize về kích thước model yêu cầu
            resized = cv2.resize(frame, (self.img_size, self.img_size))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            normalized = rgb_frame.astype(np.float32) / 255.0
            
            # Add batch dimension
            batched = np.expand_dims(normalized, axis=0)
            
            return batched
            
        except Exception as e:
            print(f"❌ Error preprocessing frame: {e}")
            return None
    
    def predict(self, frame):
        """Nhận diện chữ số từ frame"""
        if not self.is_initialized or self.model is None:
            return "Model chưa khởi tạo", 0.0
        
        try:
            # Tiền xử lý frame
            processed_frame = self.preprocess_frame(frame)
            if processed_frame is None:
                return "Lỗi xử lý ảnh", 0.0
            
            # Dự đoán
            predictions = self.model.predict(processed_frame, verbose=0)
            confidence = np.max(predictions[0])
            predicted_class = np.argmax(predictions[0])
            
            # Lấy tên class
            if predicted_class < len(self.class_names):
                prediction = self.class_names[predicted_class]
            else:
                prediction = f"Unknown_{predicted_class}"
            
            return prediction, float(confidence)
            
        except Exception as e:
            print(f"❌ Error in digit prediction: {e}")
            return "Lỗi nhận diện", 0.0
    
    def close(self):
        """Release MediaPipe resources"""
        if self.hands:
            self.hands.close()

# Global instance
digit_recognizer = DigitRecognizer()

def init_digit_recognizer(model_path=None):
    """Khởi tạo digit recognizer"""
    return digit_recognizer.initialize(model_path)

def get_digit_recognizer_instance():
    """Lấy instance của digit recognizer"""
    return digit_recognizer if digit_recognizer.is_initialized else None