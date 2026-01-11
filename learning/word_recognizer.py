# learning/word_recognizer.py
import os
import time
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import traceback
from collections import deque
import base64
from django.conf import settings

# ==================== MODEL ARCHITECTURE MATCHING asl_advanced.pth ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(mlp_hidden, dim), 
            nn.Dropout(dropout)
        )
        self.drop_path = nn.Identity()
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers, batch_first=True, 
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.cnn_dim = 512
        self.left_hand_dim = 63
        self.right_hand_dim = 63
        self.left_shape_dim = 15
        self.right_shape_dim = 15
        self.motion_dim = 18
        self.flow_dim = 1
        self.edge_dim = 6
        
        # Projection layers for each feature type
        self.cnn_proj = nn.Sequential(
            nn.Linear(self.cnn_dim, hidden_dim // 2), 
            nn.LayerNorm(hidden_dim // 2), 
            nn.GELU(), 
            nn.Dropout(dropout)
        )
        
        hand_combined_dim = self.left_hand_dim + self.right_hand_dim
        self.hand_proj = nn.Sequential(
            nn.Linear(hand_combined_dim, hidden_dim // 4), 
            nn.LayerNorm(hidden_dim // 4), 
            nn.GELU(), 
            nn.Dropout(dropout * 0.7)
        )
        
        shape_motion_dim = self.left_shape_dim + self.right_shape_dim + self.motion_dim
        self.shape_motion_proj = nn.Sequential(
            nn.Linear(shape_motion_dim, hidden_dim // 8), 
            nn.LayerNorm(hidden_dim // 8), 
            nn.GELU(), 
            nn.Dropout(dropout * 0.5)
        )
        
        edge_flow_dim = self.edge_dim + self.flow_dim
        self.edge_flow_proj = nn.Sequential(
            nn.Linear(edge_flow_dim, hidden_dim // 8), 
            nn.LayerNorm(hidden_dim // 8), 
            nn.GELU(), 
            nn.Dropout(dropout * 0.5)
        )
        
        total_proj_dim = hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 8 + hidden_dim // 8
        self.fusion = nn.Sequential(
            nn.Linear(total_proj_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.GELU(), 
            nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(
            nn.Linear(total_proj_dim, total_proj_dim), 
            nn.Sigmoid()
        )
    
    def forward(self, x):
        idx = 0
        cnn_feat = x[:, :, idx:idx+self.cnn_dim]; idx += self.cnn_dim
        left_hand = x[:, :, idx:idx+self.left_hand_dim]; idx += self.left_hand_dim
        right_hand = x[:, :, idx:idx+self.right_hand_dim]; idx += self.right_hand_dim
        hand_feat = torch.cat([left_hand, right_hand], dim=-1)
        left_shape = x[:, :, idx:idx+self.left_shape_dim]; idx += self.left_shape_dim
        right_shape = x[:, :, idx:idx+self.right_shape_dim]; idx += self.right_shape_dim
        motion = x[:, :, idx:idx+self.motion_dim]; idx += self.motion_dim
        shape_motion = torch.cat([left_shape, right_shape, motion], dim=-1)
        flow = x[:, :, idx:idx+self.flow_dim]; idx += self.flow_dim
        edge = x[:, :, idx:idx+self.edge_dim]
        edge_flow = torch.cat([edge, flow], dim=-1)
        
        cnn_out = self.cnn_proj(cnn_feat)
        hand_out = self.hand_proj(hand_feat)
        shape_motion_out = self.shape_motion_proj(shape_motion)
        edge_flow_out = self.edge_flow_proj(edge_flow)
        
        combined = torch.cat([cnn_out, hand_out, shape_motion_out, edge_flow_out], dim=-1)
        gate = self.gate(combined)
        combined = combined * gate
        fused = self.fusion(combined)
        return fused

class AdvancedASLModel(nn.Module):
    def __init__(self, input_dim=693, num_classes=100, hidden_dim=1024, num_heads=16, 
                 num_layers=6, dropout=0.4, attention_dropout=0.2):
        super().__init__()
        
        self.feature_fusion = AdaptiveFeatureFusion(hidden_dim, dropout)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.bilstm = BiLSTMEncoder(hidden_dim, hidden_dim, num_layers=3, dropout=dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads, mlp_ratio=4, dropout=attention_dropout) 
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), 
            nn.Tanh(), 
            nn.Dropout(dropout * 0.5), 
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.LayerNorm(hidden_dim // 2), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), 
            nn.LayerNorm(hidden_dim // 4), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.feature_fusion(x)
        x = self.pos_encoder(x)
        x = self.bilstm(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        attn_weights = F.softmax(self.attention_pool(x), dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        out = self.classifier(pooled)
        return out

class SimpleCNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=6, feature_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(512, feature_dim, 3, padding=1), nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d(1)
        )
        self.feature_dim = feature_dim
    
    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return x

# ==================== SIMPLIFIED FEATURE EXTRACTOR FOR WEB ====================
class SimpleFeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.img_size = 224
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        
        # Initialize CNN (for compatibility, but we'll use simpler features for web)
        self.cnn = SimpleCNNFeatureExtractor(in_channels=6, feature_dim=512)
        self.cnn = self.cnn.to(device).eval()
    
    def extract_hand_keypoints(self, frame_rgb):
        try:
            results = self.mp_hands.process(frame_rgb)
            
            if not results.multi_hand_landmarks:
                return None, None
            
            left_hand = None
            right_hand = None
            
            for hand_lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_info.classification[0].label
                kps = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32)
                
                if label == 'Left':
                    left_hand = kps
                else:
                    right_hand = kps
            
            return left_hand, right_hand
        except Exception as e:
            print(f"Hand detection error: {e}")
            return None, None
    
    def normalize_hand_keypoints(self, hand_kps):
        if hand_kps is None or np.all(hand_kps == 0):
            return np.zeros(63, dtype=np.float32)
        
        wrist = hand_kps[0].copy()
        hand_centered = hand_kps - wrist
        scale = np.max(np.abs(hand_centered[:, :2])) + 1e-6
        hand_centered[:, :2] /= scale
        return hand_centered.flatten().astype(np.float32)
    
    def extract_simple_features(self, frame):
        """Extract simplified features for web usage"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract hand keypoints
            lh_kps, rh_kps = self.extract_hand_keypoints(frame_rgb)
            lh_norm = self.normalize_hand_keypoints(lh_kps)
            rh_norm = self.normalize_hand_keypoints(rh_kps)
            
            # For web version, we'll use simpler features
            # Use zeros for other features to match expected input dimension
            cnn_feat = np.zeros(512, dtype=np.float32)
            lh_shape = np.zeros(15, dtype=np.float32)
            rh_shape = np.zeros(15, dtype=np.float32)
            motion_feat = np.zeros(18, dtype=np.float32)
            flow_mag = 0.0
            edge_stats = np.zeros(6, dtype=np.float32)
            
            # Concatenate all features to match expected input dimension (693)
            features = np.concatenate([
                cnn_feat, lh_norm, rh_norm, lh_shape, rh_shape, motion_feat, [flow_mag], edge_stats
            ]).astype(np.float32)
            
            return features, lh_kps, rh_kps
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            # Return zeros with correct dimension
            return np.zeros(693, dtype=np.float32), None, None
    
    def close(self):
        if hasattr(self, 'mp_hands'):
            self.mp_hands.close()

# ==================== MAIN RECOGNIZER CLASS ====================
class SimpleWordRecognizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")
        
        self.model = None
        self.extractor = None
        self.is_initialized = False
        self.buffer = deque(maxlen=30)  # Buffer for temporal sequence
        self.last_prediction = "Chưa nhận diện"
        self.last_confidence = 0.0
        self.prediction_history = deque(maxlen=3)
        self.class_list = []
        
    def initialize(self, model_path=None):
        """Khởi tạo model từ vựng với asl_advanced.pth"""
        try:
            print("🚀 Initializing ASL Word Recognizer with REAL model...")
            
            # Load danh sách từ vựng
            self.class_list = self.load_class_names_from_txt()
            if not self.class_list:
                print("❌ Không thể load danh sách từ vựng")
                return False
            
            print(f"📖 Loaded {len(self.class_list)} word classes")
            
            # Tìm model path
            if model_path is None:
                possible_paths = [
                    "asl_advanced.pth",
                    "../asl_advanced.pth",
                    "./asl_advanced.pth",
                    "models/asl_advanced.pth",
                    "../models/asl_advanced.pth",
                    os.path.join(os.path.dirname(__file__), "asl_advanced.pth"),
                    os.path.join(os.path.dirname(__file__), "..", "asl_advanced.pth"),
                    os.path.join(os.path.dirname(__file__), "..", "models", "asl_advanced.pth")
                ]
                model_path = next((p for p in possible_paths if os.path.exists(p)), None)
            
            if model_path is None:
                print("❌ Word model file asl_advanced.pth not found")
                return False
            
            print(f"📁 Found model at: {model_path}")
            
            # Load model thật với architecture đúng
            self.model = self.load_pretrained_model(model_path)
            if self.model is None:
                print("❌ Failed to load model")
                return False
            
            # Initialize feature extractor
            self.extractor = SimpleFeatureExtractor(self.device)
            
            self.is_initialized = True
            print("✅ ASL Word Recognizer initialized successfully with REAL MODEL!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing word recognizer: {e}")
            traceback.print_exc()
            return False
    
    def load_pretrained_model(self, model_path):
        """Load model với architecture khớp với asl_advanced.pth"""
        try:
            print(f"🔧 Loading model from {model_path}...")
            
            # Tạo model với architecture đúng
            model = AdvancedASLModel(
                input_dim=693,
                num_classes=len(self.class_list),
                hidden_dim=1024,
                num_heads=16,
                num_layers=6,
                dropout=0.4,
                attention_dropout=0.2
            )
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            print(f"📦 Checkpoint type: {type(checkpoint)}")
            
            # Extract state dict từ checkpoint
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print("✅ Loaded using 'model_state_dict'")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print("✅ Loaded using 'state_dict'")
                else:
                    state_dict = checkpoint
                    print("✅ Loaded directly from dict")
            else:
                state_dict = checkpoint
                print("✅ Loaded as direct state_dict")
            
            # Load state dict
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            print(f"✅ Model loaded successfully with {len(self.class_list)} classes")
            return model
            
        except Exception as e:
            print(f"❌ Error loading pretrained model: {e}")
            traceback.print_exc()
            return None
    
    def load_class_names_from_txt(self):
        """Load danh sách từ vựng từ file txt"""
        try:
            class_list = []
            
            # Tìm file top_100_glosses.txt
            possible_paths = [
                "top_100_glosses.txt",
                "../top_100_glosses.txt",
                "./top_100_glosses.txt",
                "models/top_100_glosses.txt",
                os.path.join(os.path.dirname(__file__), "top_100_glosses.txt"),
                os.path.join(os.path.dirname(__file__), "..", "top_100_glosses.txt")
            ]
            
            txt_path = next((p for p in possible_paths if os.path.exists(p)), None)
            
            if txt_path is None:
                print("❌ top_100_glosses.txt not found")
                return []
            
            print(f"📁 Loading word classes from: {txt_path}")
            
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        word = line.split(':', 1)[1].strip()
                        class_list.append(word)
                    elif line and not line.isspace():
                        class_list.append(line.strip())
            
            print(f"✅ Successfully loaded {len(class_list)} words")
            return class_list
            
        except Exception as e:
            print(f"❌ Error loading class names from txt: {e}")
            return []
    
    def process_frame(self, frame):
        """Xử lý frame và trả về kết quả nhận diện từ với model thật"""
        if not self.is_initialized:
            return "Model chưa khởi tạo", 0.0
        
        try:
            # Thêm frame vào buffer
            features, lh_kps, rh_kps = self.extractor.extract_simple_features(frame)
            self.buffer.append(features)
            
            # Chỉ nhận diện khi có đủ frames trong buffer
            if len(self.buffer) < 15:
                progress = len(self.buffer) / 15.0
                return f"Đang thu thập... ({len(self.buffer)}/15)", progress * 30
            
            # Tạo sequence từ buffer
            sequence_tensor = self.create_sequence_tensor()
            if sequence_tensor is None:
                return "Không đủ dữ liệu", 0.0
            
            # Nhận diện với model
            prediction, confidence = self.real_word_recognition(sequence_tensor)
            
            print(f"🎯 REAL MODEL PREDICTION: {prediction}, Confidence: {confidence:.2f}")
            
            # Làm mượt kết quả
            if confidence > 0.3:
                self.prediction_history.append((prediction, confidence))
                
                if len(self.prediction_history) >= 2:
                    votes = {}
                    for pred, conf in self.prediction_history:
                        votes[pred] = votes.get(pred, 0) + conf
                    best_pred = max(votes.items(), key=lambda x: x[1])
                    self.last_prediction = best_pred[0]
                    self.last_confidence = best_pred[1] / len(self.prediction_history)
                else:
                    self.last_prediction = prediction
                    self.last_confidence = confidence
            else:
                self.last_prediction = prediction
                self.last_confidence = confidence
            
            return self.last_prediction, self.last_confidence * 100
            
        except Exception as e:
            print(f"❌ Error processing word frame: {e}")
            traceback.print_exc()
            return "Lỗi xử lý", 0.0
    
    def create_sequence_tensor(self):
        """Tạo sequence tensor từ buffer"""
        try:
            if len(self.buffer) < 15:
                return None
            
            # Lấy 30 frames gần nhất (hoặc tất cả nếu ít hơn)
            sequence = list(self.buffer)[-30:]
            
            # Resample về 30 frames nếu cần
            if len(sequence) > 30:
                indices = np.linspace(0, len(sequence) - 1, 30, dtype=int)
                sequence = [sequence[i] for i in indices]
            elif len(sequence) < 30:
                # Pad với frame cuối cùng
                last_frame = sequence[-1]
                padding = [last_frame] * (30 - len(sequence))
                sequence.extend(padding)
            
            sequence_array = np.array(sequence)
            sequence_tensor = torch.FloatTensor(sequence_array).unsqueeze(0).to(self.device)
            
            return sequence_tensor
            
        except Exception as e:
            print(f"❌ Sequence creation error: {e}")
            return None
    
    def real_word_recognition(self, sequence_tensor):
        """Nhận diện từ với model thật"""
        try:
            with torch.no_grad():
                outputs = self.model(sequence_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                confidence = confidence.item()
                predicted_idx = predicted_idx.item()
                
                if predicted_idx < len(self.class_list):
                    prediction = self.class_list[predicted_idx]
                else:
                    prediction = f"Unknown_{predicted_idx}"
                
                print(f"🔍 Model output - Class: {predicted_idx}, Confidence: {confidence:.3f}")
                
                # Áp dụng threshold
                if confidence < 0.3:
                    return "Cử chỉ không rõ ràng", confidence
                
                return prediction, confidence
                
        except Exception as e:
            print(f"❌ Real word recognition error: {e}")
            traceback.print_exc()
            return "Lỗi nhận diện", 0.0
    
    def reset(self):
        """Reset buffer và lịch sử nhận diện"""
        self.buffer.clear()
        self.prediction_history.clear()
        self.last_prediction = "Đã reset"
        self.last_confidence = 0.0
        print("🔄 Word recognizer reset")

# Global instance
word_recognizer = SimpleWordRecognizer()

def init_word_recognizer(model_path=None):
    return word_recognizer.initialize(model_path)

def get_word_recognizer():
    return word_recognizer if word_recognizer.is_initialized else None

def get_word_recognizer_lazy(model_path=None):
    if not word_recognizer.is_initialized:
        success = word_recognizer.initialize(model_path)
        if success:
            print("🎉 Word recognizer initialized successfully with REAL MODEL!")
        else:
            print("💥 Word recognizer initialization FAILED!")
        return word_recognizer if success else None
    return word_recognizer