# ASL Learning Web Application

Ứng dụng web học ngôn ngữ ký hiệu Mỹ (American Sign Language - ASL) với tính năng nhận diện cử chỉ tay bằng AI.

## 🎯 Tính năng chính

### 1. Học Chữ Cái (Letters)
- Hiển thị hình ảnh 26 chữ cái ASL (A-Z)
- Video hướng dẫn từ Firebase Storage
- Nhận diện realtime bằng camera với model MobileNetV2

### 2. Học Chữ Số (Digits)
- Hiển thị hình ảnh 10 chữ số ASL (0-9)
- Video hướng dẫn từ Firebase Storage
- Nhận diện realtime bằng camera với model MobileNetV2

### 3. Học Từ Vựng (Words)
- Danh sách 100 từ vựng ASL phổ biến
- Video hướng dẫn từ Firebase Storage
- Nhận diện từ vựng với YOLO + Transformer model

### 4. Luyện Tập (Practice)
- **Nhận diện ký tự**: Camera realtime nhận diện chữ cái/số
- **Nhận diện từ vựng**: Camera realtime nhận diện từ ASL

## 🛠️ Công nghệ sử dụng

- **Backend**: Django 4.2.11
- **AI/ML**:
  - TensorFlow 2.15.0 + MobileNetV2 (Letters & Digits)
  - PyTorch 2.1.2 + Transformer (Words)
  - MediaPipe 0.10.14 (Hand detection & keypoints)
  - Ultralytics YOLOv8 (Hand detection for words)
  - OpenCV 4.9.0.80 (Image processing)
- **Database**: Firebase Firestore
- **Storage**: Firebase Storage
- **Frontend**: Bootstrap 5, JavaScript

## 📁 Cấu trúc dự án

```
Web/
├── asl_web/                    # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── firebase-service-account.json
├── learning/                   # Main Django app
│   ├── views.py               # Views & API endpoints
│   ├── urls.py                # URL routing
│   ├── ai_recognizer.py       # Letter/Digit recognizer
│   ├── word_recognizer_v3.py  # Word recognizer (YOLO + Transformer)
│   ├── templates/learning/    # HTML templates
│   └── static/learning/       # CSS, JS, images
├── Models/                     # AI model files
│   ├── yolov8_asl_final.h5        # Letters model
│   ├── yolov8_asl_digits_final.h5 # Digits model
│   └── asl_advanced.pth                # Words model (Transformer)
├── top_100_glosses.txt        # 100 từ vựng ASL
├── yolov8n.pt                 # YOLOv8 model for hand detection
├── manage.py
├── requirements.txt
└── Readme.md
```

## 🚀 Hướng dẫn cài đặt

### Yêu cầu hệ thống
- Python 3.10
- CUDA (khuyến nghị cho GPU acceleration)
- Webcam

### Bước 1: Clone dự án
```bash
git clone <repository-url>
cd Web
```

### Bước 2: Tạo môi trường ảo (khuyến nghị dùng Conda)
```bash
conda create -n kltn2 python=3.10
conda activate kltn2
```

### Bước 3: Cài đặt thư viện

**Cách 1: Cài đặt nhanh (khuyến nghị)**
```bash
pip install --force-reinstall protobuf==4.25.3 firebase-admin==7.1.0 google-cloud-firestore==2.21.0 google-cloud-storage==3.6.0 google-cloud-core==2.5.0 googleapis-common-protos==1.72.0 google-api-core==2.28.1 google-auth==2.41.1 grpcio==1.62.2 grpcio-status==1.62.2 mediapipe==0.10.14 numpy==1.26.4 opencv-python==4.9.0.80 tensorflow==2.15.0 torch==2.1.2 torchvision==0.16.2 Django==4.2.11 ultralytics==8.1.0
```

**Cách 2: Cài từ requirements.txt**
```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình Firebase
1. Tạo project trên Firebase Console
2. Tải file `firebase-service-account.json` từ Project Settings > Service Accounts
3. Đặt file vào thư mục `asl_web/`
4. Cấu hình Firestore database với ID `aslweb`

### Bước 5: Chạy ứng dụng
```bash
python manage.py runserver
```

Truy cập: http://127.0.0.1:8000

## 📡 API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/` | GET | Trang chủ |
| `/alphabet/` | GET | Chọn học chữ cái/số |
| `/alphabet/letters/` | GET | Học chữ cái A-Z |
| `/alphabet/digits/` | GET | Học chữ số 0-9 |
| `/words/` | GET | Học từ vựng (100 từ) |
| `/practice/` | GET | Trang luyện tập |
| `/practice/camera/` | GET | Nhận diện ký tự realtime |
| `/practice/words/` | GET | Nhận diện từ vựng realtime |
| `/api/recognize/` | POST | API nhận diện chữ cái/số |
| `/api/recognize/words/` | POST | API nhận diện từ vựng |
| `/api/letters/` | GET | Lấy ảnh chữ cái từ Firebase |
| `/api/digits/` | GET | Lấy ảnh chữ số từ Firebase |
| `/api/words/` | GET | Lấy danh sách từ + video từ Firebase |

## 🤖 Models AI

### 1. YOLOv8 - Letters (yolov8_asl_final.h5)
- Input: 224x224x3 RGB image
- Output: 26 classes (A-Z)
- Accuracy: ~95%

### 2. YOLOv8 - Digits (yolov8_asl_digits_final.h5)
- Input: 224x224x3 RGB image
- Output: 10 classes (0-9)
- Accuracy: ~97%

### 3. Transformer - Words (asl_advanced.pth)
- Input: Sequence of 120 frames (693 features/frame)
- Features: CNN + MediaPipe keypoints + Motion + Edge
- Output: 100 word classes
- Architecture: BiLSTM + Transformer + Attention Pooling

## 📋 Danh sách 100 từ vựng ASL

Xem file `top_100_glosses.txt` để biết danh sách đầy đủ các từ được hỗ trợ.

## ⚠️ Lưu ý

1. **Dependency conflicts**: Các thư viện mediapipe, firebase-admin, protobuf có thể xung đột. Sử dụng đúng phiên bản trong lệnh cài đặt.

2. **GPU**: Nếu có NVIDIA GPU, cài thêm CUDA toolkit để tăng tốc inference.

3. **Camera**: Đảm bảo browser có quyền truy cập webcam.

4. **Firebase**: Cần có file `firebase-service-account.json` hợp lệ để load dữ liệu.

## 👥 Tác giả

- Khóa luận tốt nghiệp 2025

## 📄 License

MIT License

