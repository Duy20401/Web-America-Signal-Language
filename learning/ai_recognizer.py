# learning/ai_recognizer.py
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from collections import deque
import os

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

class ASLRecognition:
    def __init__(self, model_path):
        self.is_initialized = False
        try:
            # Load model
            self.model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})
            
            # Classes bao gồm chữ cái + del, nothing, space (khớp với model)
            self.CLASSES = sorted([
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                'del', 'nothing', 'space'
            ])
            
            self.IMG_SIZE = 224
            
            # Mediapipe setup với ngưỡng thấp hơn để dễ phát hiện tay
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.3,  # Giảm từ 0.7 xuống 0.3
                min_tracking_confidence=0.3,    # Giảm từ 0.6 xuống 0.3
                static_image_mode=False,         # Tối ưu cho video stream
                model_complexity=1               # Model trung bình (0=lite, 1=full, 2=heavy)
            )
            
            # Track hand detection and smooth predictions giống script realtime
            self.last_hand_detected = False
            self.last_bbox = None
            self.prediction_buffer = deque(maxlen=5)
            
            # Motion detection
            self.prev_frame = None
            self.min_motion_area = 500  # Diện tích tối thiểu để coi là motion

            self.is_initialized = True
            print("✅ ASL Recognition model loaded successfully!")
        except Exception as e:
            self.is_initialized = False
            print(f"❌ Failed to initialize ASL Recognition: {e}")
            raise
    
    def process_frame(self, frame):
        """
        Xử lý frame và nhận diện ký hiệu
        Trả về: (prediction, confidence, bbox)
        """
        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.hands.process(rgb)

            self.last_hand_detected = False

            if not results.multi_hand_landmarks:
                self.prediction_buffer.clear()
                self.last_bbox = None
                print("👋 No hand detected in frame")
                return "--", 0.0, None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get hand bounding box
                    h, w, _ = frame.shape
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    x_min = int(min(x_coords) * w) - 20
                    x_max = int(max(x_coords) * w) + 20
                    y_min = int(min(y_coords) * h) - 20
                    y_max = int(max(y_coords) * h) + 20
                    
                    # Ensure within frame bounds
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w, x_max), min(h, y_max)
                    
                    # Extract ROI
                    roi = frame[y_min:y_max, x_min:x_max]
                    if roi.size == 0:
                        continue
                    
                    # Preprocess image
                    img = cv2.resize(roi, (self.IMG_SIZE, self.IMG_SIZE))
                    img = img.astype("float32") / 255.0
                    img = np.expand_dims(img, axis=0)

                    # Predict
                    preds = self.model.predict(img, verbose=0)
                    class_id = np.argmax(preds)
                    confidence = float(np.max(preds))
                    prediction = self.CLASSES[class_id]

                    self.last_hand_detected = True
                    self.last_bbox = (x_min, y_min, x_max, y_max)
                    self.prediction_buffer.append((prediction, confidence))

                    smooth_pred, smooth_conf = self._get_smoothed_prediction()

                    # Trả về prediction gốc từ model (bao gồm cả 'del', 'nothing', 'space')
                    # Backend sẽ xử lý filter nếu cần
                    return smooth_pred, smooth_conf, self.last_bbox

                # Nếu không return bên trong vòng lặp (ví dụ ROI trống)
                self.last_hand_detected = False
                self.last_bbox = None
                return "--", 0.0, None
            
        except Exception as e:
            print(f"❌ Lỗi trong process_frame: {e}")
            return "Lỗi nhận diện", 0.0, None
    
    def predict(self, frame):
        """
        Predict với motion detection để crop ROI động
        Trả về: (prediction, confidence)
        """
        if not self.is_initialized or self.model is None:
            return "Model chưa khởi tạo", 0.0
        
        try:
            # Preprocess toàn bộ frame
            resized = cv2.resize(frame, (self.IMG_SIZE, self.IMG_SIZE))
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb_frame.astype(np.float32) / 255.0
            batched = np.expand_dims(normalized, axis=0)
            
            # Motion detection để crop ROI
            predict_frame = batched  # Default: predict full frame
            
            if self.prev_frame is not None:
                # Tính difference
                diff = cv2.absdiff(self.prev_frame, resized)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Lấy contour lớn nhất
                    c = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(c)
                    
                    if area > self.min_motion_area:
                        # Crop ROI từ motion area
                        x, y, w, h = cv2.boundingRect(c)
                        # Đảm bảo trong bounds
                        x1, y1 = max(0, x-10), max(0, y-10)
                        x2, y2 = min(self.IMG_SIZE, x+w+10), min(self.IMG_SIZE, y+h+10)
                        
                        roi = resized[y1:y2, x1:x2]
                        if roi.size > 0:
                            # Resize ROI về IMG_SIZE
                            roi_resized = cv2.resize(roi, (self.IMG_SIZE, self.IMG_SIZE))
                            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                            roi_normalized = roi_rgb.astype(np.float32) / 255.0
                            predict_frame = np.expand_dims(roi_normalized, axis=0)
                            print(f"🎯 Motion detected, using ROI: {x1},{y1} to {x2},{y2}")
            
            # Update previous frame
            self.prev_frame = resized.copy()
            
            # Dự đoán
            predictions = self.model.predict(predict_frame, verbose=0)
            confidence = np.max(predictions[0])
            predicted_class = np.argmax(predictions[0])
            
            # Lấy tên class
            if predicted_class < len(self.CLASSES):
                prediction = self.CLASSES[predicted_class]
                # Chỉ trả về chữ cái A-Z, bỏ del, nothing, space
                if prediction.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    return prediction.upper(), float(confidence)
                else:
                    return "--", 0.0
            else:
                return "--", 0.0
            
        except Exception as e:
            print(f"❌ Error in ASL predict: {e}")
            return "Lỗi nhận diện", 0.0
    
    def recognize_frame(self, frame):
        """
        Method cũ để tương thích ngược
        Trả về dict như cũ
        """
        prediction, confidence, bbox = self.process_frame(frame)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'bbox': bbox
        }

    def _get_smoothed_prediction(self):
        """Làm mượt kết quả dựa trên buffer gần nhất"""
        if not self.prediction_buffer:
            return "--", 0.0

        labels = [label for label, _ in self.prediction_buffer]
        dominant = max(set(labels), key=labels.count)
        confidences = [conf for label, conf in self.prediction_buffer if label == dominant]

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return dominant, float(avg_conf)
    
    def close(self):
        """Giải phóng tài nguyên"""
        self.hands.close()

# Global instance
asl_recognizer = None

def init_recognizer(model_path):
    """Khởi tạo recognizer toàn cục"""
    global asl_recognizer
    try:
        asl_recognizer = ASLRecognition(model_path)
        return asl_recognizer.is_initialized
    except Exception as e:
        print(f"❌ Lỗi khởi tạo recognizer: {e}")
        asl_recognizer = None
        return False

def get_recognizer():
    """Lấy instance của recognizer"""
    if asl_recognizer and asl_recognizer.is_initialized:
        return asl_recognizer
    return None