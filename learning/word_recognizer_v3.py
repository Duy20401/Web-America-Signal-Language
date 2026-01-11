# learning/word_recognizer_v3.py
"""
ASL Word Recognizer V3 - Based on asl_desktopV3.py
Features: YOLO hand detection, MediaPipe keypoints, Kalman filtering, 
6-channel CNN, motion features, BiLSTM + Transformer model
"""

import os
import time
import json
import traceback
from collections import deque
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import euclidean

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '2'

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None


# ==================== MODEL ARCHITECTURE ====================
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
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim), nn.Dropout(dropout)
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
        
        self.cnn_proj = nn.Sequential(
            nn.Linear(self.cnn_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout)
        )
        hand_combined_dim = self.left_hand_dim + self.right_hand_dim
        self.hand_proj = nn.Sequential(
            nn.Linear(hand_combined_dim, hidden_dim // 4), nn.LayerNorm(hidden_dim // 4),
            nn.GELU(), nn.Dropout(dropout * 0.7)
        )
        shape_motion_dim = self.left_shape_dim + self.right_shape_dim + self.motion_dim
        self.shape_motion_proj = nn.Sequential(
            nn.Linear(shape_motion_dim, hidden_dim // 8), nn.LayerNorm(hidden_dim // 8),
            nn.GELU(), nn.Dropout(dropout * 0.5)
        )
        edge_flow_dim = self.edge_dim + self.flow_dim
        self.edge_flow_proj = nn.Sequential(
            nn.Linear(edge_flow_dim, hidden_dim // 8), nn.LayerNorm(hidden_dim // 8),
            nn.GELU(), nn.Dropout(dropout * 0.5)
        )
        total_proj_dim = hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 8 + hidden_dim // 8
        self.fusion = nn.Sequential(
            nn.Linear(total_proj_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(nn.Linear(total_proj_dim, total_proj_dim), nn.Sigmoid())
    
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
    def __init__(self, input_dim, num_classes, hidden_dim=1024, num_heads=16,
                 num_layers=6, dropout=0.4, attention_dropout=0.2):
        super().__init__()
        expected_dim = 512 + 63 + 63 + 15 + 15 + 18 + 1 + 6  # = 693
        assert input_dim == expected_dim, f"Expected {expected_dim}, got {input_dim}"
        
        self.feature_fusion = AdaptiveFeatureFusion(hidden_dim, dropout)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.bilstm = BiLSTMEncoder(hidden_dim, hidden_dim, num_layers=3, dropout=dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads, mlp_ratio=4, dropout=attention_dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.Tanh(),
            nn.Dropout(dropout * 0.5), nn.Linear(hidden_dim // 4, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.LayerNorm(hidden_dim // 4), nn.GELU(), nn.Dropout(dropout),
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


# ==================== KALMAN FILTER ====================
class HandKeypointsKalman:
    """Kalman filter for 21 hand keypoints"""
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.filters = []
        for _ in range(21):
            self.filters.append({
                'x': None, 'P': None,
                'Q': np.eye(4, dtype=np.float32) * process_noise,
                'R': np.eye(2, dtype=np.float32) * measurement_noise,
                'F': np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
                'H': np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
            })
    
    def reset(self):
        for f in self.filters:
            f['x'] = None
            f['P'] = None
    
    def update(self, keypoints):
        if keypoints is None or np.all(keypoints == 0):
            return keypoints
        smoothed = keypoints.copy()
        for i in range(21):
            z = keypoints[i, :2].astype(np.float32)
            f = self.filters[i]
            if f['x'] is None:
                f['x'] = np.array([z[0], z[1], 0, 0], dtype=np.float32)
                f['P'] = np.eye(4, dtype=np.float32)
                smoothed[i, :2] = z
                continue
            x_pred = f['F'] @ f['x']
            P_pred = f['F'] @ f['P'] @ f['F'].T + f['Q']
            y = z - f['H'] @ x_pred
            S = f['H'] @ P_pred @ f['H'].T + f['R']
            try:
                K = P_pred @ f['H'].T @ np.linalg.inv(S)
            except:
                K = np.zeros((4, 2), dtype=np.float32)
            f['x'] = x_pred + K @ y
            f['P'] = (np.eye(4, dtype=np.float32) - K @ f['H']) @ P_pred
            smoothed[i, :2] = f['x'][:2]
        return smoothed


# ==================== FEATURE EXTRACTOR ====================
class ConsistentFeatureExtractor:
    """Feature extractor matching asl_desktopV3.py"""
    
    def __init__(self, device, debug=False):
        self.device = device
        self.img_size = 224
        self.debug = debug
        
        # YOLO
        self.yolo = None
        if YOLO_AVAILABLE:
            try:
                self.yolo = YOLO('yolov8n.pt')
                self.yolo.fuse()
                print("✅ YOLO loaded")
            except Exception as e:
                print(f"⚠️ YOLO load failed: {e}")
        
        # CNN
        self.cnn = SimpleCNNFeatureExtractor(in_channels=6, feature_dim=512)
        self.cnn = self.cnn.to(device).eval()
        
        # MediaPipe
        self.mp_hands = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=2, model_complexity=0,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
        
        # Kalman filters
        self.kalman_lh = HandKeypointsKalman()
        self.kalman_rh = HandKeypointsKalman()
        
        self.prev_gray = None
        self.frame_count = 0
        self.hand_detect_count = 0
    
    def detect_hands_yolo(self, frame):
        """Detect hand regions with YOLO"""
        if self.yolo is None:
            h, w = frame.shape[:2]
            return [[0, 0, w, h, 1.0]]
        try:
            results = self.yolo(frame, verbose=False, conf=0.25, classes=[0])
            boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    boxes.append([int(x1), int(y1), int(x2), int(y2), conf])
            if len(boxes) == 0:
                h, w = frame.shape[:2]
                boxes = [[0, 0, w, h, 1.0]]
            return boxes[:2]
        except:
            h, w = frame.shape[:2]
            return [[0, 0, w, h, 1.0]]
    
    def expand_bbox(self, bbox, margin=0.2, shape=None):
        x1, y1, x2, y2, conf = bbox
        w, h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - w * margin))
        y1 = max(0, int(y1 - h * margin))
        x2 = min(shape[1] if shape else 99999, int(x2 + w * margin))
        y2 = min(shape[0] if shape else 99999, int(y2 + h * margin))
        return [x1, y1, x2, y2, conf]
    
    def extract_edge_features(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        gray = gray.astype(np.uint8)
        try:
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            sobel = np.clip(sobel, 0, 255).astype(np.uint8)
            canny = cv2.Canny(gray, 50, 150)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)
            edges = np.stack([sobel, canny, laplacian], axis=-1)
            return edges
        except:
            return np.zeros((*gray.shape, 3), dtype=np.uint8)
    
    def extract_hand_keypoints(self, frame_rgb, bbox=None):
        if self.mp_hands is None:
            return None, None
        h, w = frame_rgb.shape[:2]
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return None, None
            roi = frame_rgb[y1:y2, x1:x2]
            if roi.size == 0:
                return None, None
        else:
            roi = frame_rgb
            x1, y1 = 0, 0
        try:
            results = self.mp_hands.process(roi)
            if not results.multi_hand_landmarks:
                return None, None
            self.hand_detect_count += 1
            left_hand, right_hand = None, None
            for hand_lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_info.classification[0].label
                kps = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32)
                roi_h, roi_w = roi.shape[:2]
                kps[:, 0] = (kps[:, 0] * roi_w + x1) / w
                kps[:, 1] = (kps[:, 1] * roi_h + y1) / h
                if label == 'Left':
                    left_hand = kps
                else:
                    right_hand = kps
            return left_hand, right_hand
        except:
            return None, None
    
    def normalize_hand_keypoints(self, hand_kps):
        if hand_kps is None or np.all(hand_kps == 0):
            return np.zeros(63, dtype=np.float32)
        wrist = hand_kps[0].copy()
        hand_centered = hand_kps - wrist
        scale = np.max(np.abs(hand_centered[:, :2])) + 1e-6
        hand_centered[:, :2] /= scale
        return hand_centered.flatten().astype(np.float32)
    
    def extract_hand_shape(self, hand_kps):
        try:
            if hand_kps is None or np.all(hand_kps == 0):
                return np.zeros(15, dtype=np.float32)
            feats = []
            tips = [4, 8, 12, 16, 20]
            wrist = hand_kps[0][:2]
            for tip_idx in tips:
                try:
                    dist = euclidean(hand_kps[tip_idx][:2], wrist)
                    feats.append(float(dist))
                except:
                    feats.append(0.0)
            for i in range(len(tips)-1):
                try:
                    dist = euclidean(hand_kps[tips[i]][:2], hand_kps[tips[i+1]][:2])
                    feats.append(float(dist))
                except:
                    feats.append(0.0)
            try:
                palm_width = euclidean(hand_kps[5][:2], hand_kps[17][:2])
                feats.append(float(palm_width))
            except:
                feats.append(0.0)
            try:
                palm_center = hand_kps[[0, 5, 9, 13, 17]][:, :2].mean(axis=0)
                openness = np.mean([euclidean(hand_kps[t][:2], palm_center) for t in tips])
                feats.append(float(openness))
            except:
                feats.append(0.0)
            for i in range(1, 5):
                base = 5 + (i-1)*4
                mid = base + 2
                tip = base + 3
                try:
                    v1 = hand_kps[mid][:2] - hand_kps[base][:2]
                    v2 = hand_kps[tip][:2] - hand_kps[mid][:2]
                    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                    feats.append(float(abs(angle)))
                except:
                    feats.append(0.0)
            if len(feats) != 15:
                return np.zeros(15, dtype=np.float32)
            return np.array(feats, dtype=np.float32)
        except:
            return np.zeros(15, dtype=np.float32)
    
    def compute_optical_flow(self, curr_gray):
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return 0.0
        try:
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
            self.prev_gray = curr_gray
            return float(mag)
        except:
            self.prev_gray = curr_gray
            return 0.0
    
    def extract_frame_features(self, frame):
        """Extract features consistently with preprocessing"""
        self.frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # YOLO detection
        hand_boxes = self.detect_hands_yolo(frame)
        if hand_boxes:
            main_bbox = self.expand_bbox(hand_boxes[0], 0.2, frame.shape)
        else:
            main_bbox = [0, 0, w, h, 1.0]
        
        # Extract ROI
        x1, y1, x2, y2 = map(int, main_bbox[:4])
        roi = frame_rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if roi.size == 0:
            roi = frame_rgb
            main_bbox = [0, 0, w, h, 1.0]
        
        # Edge features
        edges = self.extract_edge_features(roi)
        
        # Resize
        roi_resized = cv2.resize(roi, (self.img_size, self.img_size))
        edges_resized = cv2.resize(edges, (self.img_size, self.img_size))
        
        # 6-channel CNN
        frame_6ch = np.concatenate([roi_resized, edges_resized], axis=-1)
        frame_6ch_tensor = torch.from_numpy(frame_6ch / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            cnn_feat = self.cnn(frame_6ch_tensor).cpu().numpy().flatten()
        
        # Edge stats
        edge_mean = edges_resized.mean(axis=(0, 1))
        edge_std = edges_resized.std(axis=(0, 1))
        edge_stats = np.concatenate([edge_mean, edge_std]).astype(np.float32)
        
        # Hand keypoints from ROI
        lh_kps, rh_kps = self.extract_hand_keypoints(roi, None)
        
        # Kalman filtering
        if lh_kps is not None:
            lh_kps = self.kalman_lh.update(lh_kps)
        else:
            lh_kps = np.zeros((21, 3), dtype=np.float32)
        if rh_kps is not None:
            rh_kps = self.kalman_rh.update(rh_kps)
        else:
            rh_kps = np.zeros((21, 3), dtype=np.float32)
        
        # Normalize
        lh_norm = self.normalize_hand_keypoints(lh_kps)
        rh_norm = self.normalize_hand_keypoints(rh_kps)
        
        # Shape features
        lh_shape = self.extract_hand_shape(lh_kps)
        rh_shape = self.extract_hand_shape(rh_kps)
        
        # Optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_mag = self.compute_optical_flow(gray)
        
        # Motion placeholder
        motion_feat = np.zeros(18, dtype=np.float32)
        
        # Concatenate: cnn(512) + lh_norm(63) + rh_norm(63) + lh_shape(15) + rh_shape(15) + motion(18) + flow(1) + edge(6) = 693
        features = np.concatenate([
            cnn_feat, lh_norm, rh_norm, lh_shape, rh_shape, motion_feat, [flow_mag], edge_stats
        ]).astype(np.float32)
        
        return features, lh_kps, rh_kps, main_bbox
    
    def reset(self):
        self.kalman_lh.reset()
        self.kalman_rh.reset()
        self.prev_gray = None
        self.frame_count = 0
        self.hand_detect_count = 0
    
    def close(self):
        if self.mp_hands:
            self.mp_hands.close()


# ==================== MOTION COMPUTER ====================
class MotionComputer:
    @staticmethod
    def compute_motion_features(lh_seq, rh_seq):
        n = len(lh_seq)
        motion = np.zeros((n, 18), dtype=np.float32)
        for i in range(n):
            lh_wrist = lh_seq[i][0][:2]
            rh_wrist = rh_seq[i][0][:2]
            if i > 0:
                lh_prev = lh_seq[i-1][0][:2]
                rh_prev = rh_seq[i-1][0][:2]
                lh_vel = euclidean(lh_wrist, lh_prev)
                rh_vel = euclidean(rh_wrist, rh_prev)
            else:
                lh_vel = rh_vel = 0.0
            if i > 1:
                lh_prev2 = lh_seq[i-2][0][:2]
                rh_prev2 = rh_seq[i-2][0][:2]
                lh_vel_prev = euclidean(lh_seq[i-1][0][:2], lh_prev2)
                rh_vel_prev = euclidean(rh_seq[i-1][0][:2], rh_prev2)
                lh_acc = abs(lh_vel - lh_vel_prev)
                rh_acc = abs(rh_vel - rh_vel_prev)
            else:
                lh_acc = rh_acc = 0.0
            hands_dist = euclidean(lh_wrist, rh_wrist)
            if i > 0:
                lh_tips = lh_seq[i][[4, 8, 12, 16, 20]][:, :2]
                lh_tips_prev = lh_seq[i-1][[4, 8, 12, 16, 20]][:, :2]
                lh_ang_vel = np.mean([euclidean(lh_tips[j], lh_tips_prev[j]) for j in range(5)])
                rh_tips = rh_seq[i][[4, 8, 12, 16, 20]][:, :2]
                rh_tips_prev = rh_seq[i-1][[4, 8, 12, 16, 20]][:, :2]
                rh_ang_vel = np.mean([euclidean(rh_tips[j], rh_tips_prev[j]) for j in range(5)])
            else:
                lh_ang_vel = rh_ang_vel = 0.0
            if i > 0:
                lh_prev = lh_seq[i-1][0][:2]
                rh_prev = rh_seq[i-1][0][:2]
                lh_dx = lh_wrist[0] - lh_prev[0]
                lh_dy = lh_wrist[1] - lh_prev[1]
                lh_angle = np.arctan2(lh_dy, lh_dx)
                rh_dx = rh_wrist[0] - rh_prev[0]
                rh_dy = rh_wrist[1] - rh_prev[1]
                rh_angle = np.arctan2(rh_dy, rh_dx)
            else:
                lh_angle = rh_angle = 0.0
            lh_z_avg = np.mean([lh_seq[i][j][2] for j in range(21)])
            motion[i] = [
                lh_vel, rh_vel, lh_acc, rh_acc, hands_dist,
                lh_wrist[0], lh_wrist[1], rh_wrist[0], rh_wrist[1],
                lh_vel + rh_vel, abs(lh_vel - rh_vel), max(lh_vel, rh_vel),
                (lh_acc + rh_acc) / 2, lh_ang_vel, rh_ang_vel, lh_angle, rh_angle, lh_z_avg,
            ]
        return motion


# ==================== BUFFER MANAGER ====================
class FeatureBufferManager:
    def __init__(self, target_frames=120, min_frames=60):
        self.target_frames = target_frames
        self.min_frames = min_frames
        self.feature_buffer = deque(maxlen=target_frames * 2)
        self.lh_kps_buffer = deque(maxlen=target_frames * 2)
        self.rh_kps_buffer = deque(maxlen=target_frames * 2)
        self.frame_count = 0
    
    def add_frame(self, features, lh_kps, rh_kps):
        self.feature_buffer.append(features.copy())
        self.lh_kps_buffer.append(lh_kps.copy() if lh_kps is not None else np.zeros((21, 3), dtype=np.float32))
        self.rh_kps_buffer.append(rh_kps.copy() if rh_kps is not None else np.zeros((21, 3), dtype=np.float32))
        self.frame_count += 1
    
    def can_predict(self):
        return len(self.feature_buffer) >= self.min_frames
    
    def get_sequence(self):
        if not self.can_predict():
            return None
        features = list(self.feature_buffer)
        lh_kps = list(self.lh_kps_buffer)
        rh_kps = list(self.rh_kps_buffer)
        n = len(features)
        if n >= self.target_frames:
            indices = np.linspace(0, n - 1, self.target_frames, dtype=int)
        else:
            indices = np.arange(n)
        sampled_features = [features[i] for i in indices]
        sampled_lh = [lh_kps[i] for i in indices]
        sampled_rh = [rh_kps[i] for i in indices]
        motion_features = MotionComputer.compute_motion_features(sampled_lh, sampled_rh)
        motion_idx = 512 + 63 + 63 + 15 + 15
        for i in range(len(sampled_features)):
            sampled_features[i][motion_idx:motion_idx+18] = motion_features[i]
        while len(sampled_features) < self.target_frames:
            sampled_features.append(sampled_features[-1].copy())
        sequence = np.array(sampled_features[:self.target_frames], dtype=np.float32)
        return sequence
    
    def reset(self):
        self.feature_buffer.clear()
        self.lh_kps_buffer.clear()
        self.rh_kps_buffer.clear()
        self.frame_count = 0


# ==================== MAIN RECOGNIZER CLASS ====================
class WordRecognizerV3:
    """ASL Word Recognizer V3 for web usage"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ WordRecognizerV3 using device: {self.device}")
        
        self.model = None
        self.extractor = None
        self.buffer = None
        self.is_initialized = False
        self.class_list = []
        self.prediction_history = deque(maxlen=5)
        self.last_prediction = "Waiting..."
        self.last_confidence = 0.0
    
    def initialize(self, model_path=None):
        try:
            print("🚀 Initializing WordRecognizerV3...")
            
            # Load class names
            self.class_list = self._load_class_names()
            if not self.class_list:
                print("⚠️ No class names found, using defaults")
                self.class_list = [f"Class_{i}" for i in range(100)]
            print(f"📖 Loaded {len(self.class_list)} word classes")
            
            # Find model
            if model_path is None:
                possible_paths = [
                    "asl_advanced.pth", "Models/asl_advanced.pth",
                    "../asl_advanced.pth", "../Models/asl_advanced.pth",
                    os.path.join(os.path.dirname(__file__), "..", "asl_advanced.pth"),
                    os.path.join(os.path.dirname(__file__), "..", "Models", "asl_advanced.pth"),
                ]
                model_path = next((p for p in possible_paths if os.path.exists(p)), None)
            
            if model_path is None:
                print("❌ Model file not found")
                return False
            
            print(f"📁 Loading model from: {model_path}")
            
            # Load model
            self.model = self._load_model(model_path)
            if self.model is None:
                return False
            
            # Initialize extractor and buffer
            self.extractor = ConsistentFeatureExtractor(self.device)
            self.buffer = FeatureBufferManager(target_frames=120, min_frames=15)
            
            self.is_initialized = True
            print("✅ WordRecognizerV3 initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            traceback.print_exc()
            return False
    
    def _load_class_names(self):
        paths = [
            "top_100_glosses.txt", "../top_100_glosses.txt",
            "data/top_100_glosses.txt", "data/gloss_to_label.json",
            os.path.join(os.path.dirname(__file__), "..", "top_100_glosses.txt"),
        ]
        for path in paths:
            if os.path.exists(path):
                try:
                    if path.endswith('.json'):
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                sorted_items = sorted(data.items(), key=lambda x: x[1])
                                return [k for k, v in sorted_items]
                            return data
                    else:
                        with open(path, "r", encoding="utf-8") as f:
                            lines = [l.strip() for l in f if l.strip()]
                            glosses = []
                            for line in lines:
                                if ':' in line:
                                    parts = line.split(':', 1)
                                    if len(parts) == 2:
                                        glosses.append(parts[1].strip())
                                else:
                                    glosses.append(line)
                            return glosses
                except:
                    continue
        return None
    
    def _load_model(self, model_path):
        try:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict):
                state = ckpt.get("model_state_dict", ckpt)
                config = ckpt.get("args", {})
            else:
                state = ckpt
                config = {}
            
            input_dim = config.get("input_dim", 693)
            num_classes = len(self.class_list)
            hidden_dim = config.get("hidden_dim", 1024)
            num_heads = config.get("num_heads", 16)
            num_layers = config.get("num_layers", 6)
            dropout = config.get("dropout", 0.4)
            attention_dropout = config.get("attention_dropout", 0.2)
            
            model = AdvancedASLModel(
                input_dim=input_dim, num_classes=num_classes,
                hidden_dim=hidden_dim, num_heads=num_heads,
                num_layers=num_layers, dropout=dropout,
                attention_dropout=attention_dropout
            ).to(self.device)
            
            model.load_state_dict(state)
            model.eval()
            print(f"✅ Model loaded: {num_classes} classes")
            return model
        except Exception as e:
            print(f"❌ Model load error: {e}")
            traceback.print_exc()
            return None
    
    def process_frame(self, frame):
        """Process single frame and return prediction"""
        if not self.is_initialized:
            return "Model chưa khởi tạo", 0.0
        
        try:
            # Extract features
            features, lh_kps, rh_kps, bbox = self.extractor.extract_frame_features(frame)
            self.buffer.add_frame(features, lh_kps, rh_kps)
            
            # Check if we have enough frames
            buffer_size = len(self.buffer.feature_buffer)
            if buffer_size < self.buffer.min_frames:
                progress = buffer_size / self.buffer.min_frames
                return f"Đang thu thập... ({buffer_size}/{self.buffer.min_frames})", progress * 30
            
            # Get sequence and predict
            sequence = self.buffer.get_sequence()
            if sequence is None:
                return "Không đủ dữ liệu", 0.0
            
            # Predict
            x = torch.from_numpy(sequence).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                out = self.model(x)
                probs = torch.softmax(out, dim=1)
                conf, idx = torch.max(probs, dim=1)
                pred_idx = idx.item()
                pred_conf = float(conf.item())
                
                if pred_idx < len(self.class_list):
                    pred_class = self.class_list[pred_idx]
                else:
                    pred_class = f"Class_{pred_idx}"
            
            # Smooth predictions
            self.prediction_history.append((pred_class, pred_conf))
            if len(self.prediction_history) >= 2:
                votes = {}
                for p, c in self.prediction_history:
                    votes[p] = votes.get(p, 0) + c
                best_pred = max(votes.items(), key=lambda x: x[1])
                self.last_prediction = best_pred[0]
                self.last_confidence = best_pred[1] / len(self.prediction_history)
            else:
                self.last_prediction = pred_class
                self.last_confidence = pred_conf
            
            if self.last_confidence < 0.3:
                return "Cử chỉ không rõ ràng", self.last_confidence * 100
            
            return self.last_prediction, self.last_confidence * 100
            
        except Exception as e:
            print(f"❌ Process frame error: {e}")
            traceback.print_exc()
            return "Lỗi xử lý", 0.0
    
    def reset(self):
        if self.buffer:
            self.buffer.reset()
        if self.extractor:
            self.extractor.reset()
        self.prediction_history.clear()
        self.last_prediction = "Waiting..."
        self.last_confidence = 0.0
        print("🔄 WordRecognizerV3 reset")
    
    def close(self):
        if self.extractor:
            self.extractor.close()


# Global instance
word_recognizer_v3 = WordRecognizerV3()


def init_word_recognizer_v3(model_path=None):
    return word_recognizer_v3.initialize(model_path)


def get_word_recognizer_v3():
    return word_recognizer_v3 if word_recognizer_v3.is_initialized else None
