from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
from django.http import JsonResponse
import os
import base64
import cv2
import numpy as np
import time
import traceback  # THIẾU IMPORT NÀY

# Import AI recognizers
from .ai_recognizer import init_recognizer, get_recognizer
from .word_recognizer import init_word_recognizer, get_word_recognizer
from .ai_loader import get_asl_recognizer, get_word_recognizer_lazy
from .ai_loader import get_digit_recognizer

# Import Word Recognizer V3 (asl_desktopV3.py based)
try:
    from .word_recognizer_v3 import get_word_recognizer_v3, init_word_recognizer_v3
    WORD_RECOGNIZER_V3_AVAILABLE = True
except ImportError:
    WORD_RECOGNIZER_V3_AVAILABLE = False
    get_word_recognizer_v3 = None
    init_word_recognizer_v3 = None

# Firebase admin imports for server-side signed URLs
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from google.cloud import storage as gcs
    from google.oauth2 import service_account as ga_service_account
    from datetime import timedelta
except Exception:
    firebase_admin = None
# Paths to model files (used by lazy loader)
ASL_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'Models', 'yolov8_asl_final.h5')
WORD_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'Models', 'asl_advanced.pth')
DIGIT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'Models', 'yolov8_asl_digits_final.h5')

def home(request):
    """Trang chủ"""
    return render(request, 'learning/home.html')

def learn_alphabet(request):
    """Trang chọn học chữ cái hoặc chữ số"""
    return render(request, 'learning/alphabet_digits_choice.html')

@ensure_csrf_cookie
def learn_letters(request):
    """Trang học bảng chữ cái"""
    return render(request, 'learning/letters.html')

@ensure_csrf_cookie
def learn_digits(request):
    """Trang học chữ số"""
    return render(request, 'learning/digits.html')

def alphabet_detail(request, letter):
    """Chi tiết chữ cái - lấy dữ liệu từ Firestore"""
    from urllib.parse import unquote
    
    letter_upper = letter.upper()
    letter_data = {
        'letter': letter_upper,
        'image': None,
        'description': f'Học ký hiệu tay cho chữ {letter_upper} trong ngôn ngữ ký hiệu ASL.'
    }
    
    # Try to fetch from Firestore if available
    if firebase_admin and firebase_admin._apps:
        try:
            db = firestore.client(database_id='aslweb')
            # Get all documents and find one whose URL contains the letter name
            docs = db.collection('Vocabulary').stream()
            
            for doc in docs:
                data = doc.to_dict() or {}
                # Check numeric keys (0, 1, 2...) for URLs
                for key in sorted([k for k in data.keys() if k.isdigit()], key=int):
                    val = data[key]
                    if isinstance(val, str) and val.startswith('http'):
                        # Decode URL to handle %2F -> /
                        decoded_url = unquote(val).upper()
                        # Match patterns: /Alphabet/A., /Letters/A., etc.
                        if ('/ALPHABET/' in decoded_url or '/LETTERS/' in decoded_url) and f'/{letter_upper}.' in decoded_url:
                            letter_data['image'] = {'url': val}
                            break
                
                if letter_data['image']:
                    break
            
        except Exception as e:
            print(f"Error fetching letter from Firestore: {e}")
    
    context = {'letter': letter_data}
    return render(request, 'learning/alphabet_detail.html', context)

def learn_words(request):
    """Trang học từ vựng"""
    return render(request, 'learning/words.html')

def word_detail(request, word):
    """Chi tiết từ vựng"""
    context = {'word': word}
    return render(request, 'learning/word_detail.html', context)

def practice(request):
    """Trang luyện tập chính"""
    return render(request, 'learning/practice.html')

def practice_words(request):
    """Trang trung gian trước khi vào nhận diện từ vựng bằng camera"""
    return render(request, 'learning/practice_words.html')

@ensure_csrf_cookie
def practice_words_v2(request):
    """Trang luyện tập từ vựng mới với word recognizer V3"""
    model_ready = False
    if WORD_RECOGNIZER_V3_AVAILABLE:
        recognizer = get_word_recognizer_v3()
        if recognizer is None:
            # Try to initialize
            init_word_recognizer_v3(WORD_MODEL_PATH)
            recognizer = get_word_recognizer_v3()
        model_ready = recognizer is not None and recognizer.is_initialized
    return render(request, 'learning/practice_words_v2.html', {
        'model_ready': model_ready
    })

def practice_camera(request):
    """Trang luyện tập với camera - SỬ DỤNG AI THẬT"""
    # Kiểm tra cả hai model
    asl_recognizer = get_asl_recognizer(ASL_MODEL_PATH)
    digit_recognizer = get_digit_recognizer(DIGIT_MODEL_PATH)
    
    asl_ready = asl_recognizer is not None and asl_recognizer.is_initialized
    digit_ready = digit_recognizer is not None and digit_recognizer.is_initialized
    
    # Model được coi là ready nếu ít nhất một model hoạt động
    model_ready = asl_ready or digit_ready
    
    context = {
        'model_ready': model_ready,
        'asl_ready': asl_ready,
        'digit_ready': digit_ready,
    }
    return render(request, 'learning/practice_camera.html', context)

def practice_words_camera(request):
    recognizer = get_word_recognizer_lazy(WORD_MODEL_PATH)
    return render(request, 'learning/practice_words_camera.html', {
        'model_ready': recognizer is not None
    })


def api_letters_signed_urls(request):
    """Return JSON list of letters with image URLs (signed or direct).

    Reads Firestore collection `Vocabulary` and returns items: {id, name, url}.
    Requires service account JSON at `asl_web/firebase-service-account.json`.
    """
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)

    if firebase_admin is None:
        return JsonResponse({'success': False, 'error': 'firebase-admin or google-cloud-storage not installed'}, status=500)

    try:
        # initialize firebase-admin if needed
        if not firebase_admin._apps:
            sa_path = os.path.join(os.path.dirname(__file__), '..', 'asl_web', 'firebase-service-account.json')
            if not os.path.exists(sa_path):
                return JsonResponse({'success': False, 'error': f'Service account not found: {sa_path}'}, status=500)
            cred = credentials.Certificate(sa_path)
            firebase_admin.initialize_app(cred)

        db = firestore.client(database_id='aslweb')
        items = []
        
        # Read specific document for letters
        doc_ref = db.collection('Vocabulary').document('UG8NXAPDdE23fMzgJSon')
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict() or {}
            print(f"Letters document keys: {list(data.keys())}")
            # Iterate through all fields - handle both 'A:' and 'A' formats
            for key, val in data.items():
                if isinstance(val, str) and val.startswith('http'):
                    # Remove trailing colon if present
                    clean_key = key.rstrip(':')
                    items.append({'id': clean_key, 'name': clean_key.upper(), 'url': val})
                    print(f"Added letter: {clean_key} -> {val[:50]}...")
        else:
            print("Letters document does not exist!")
        
        print(f"Total letters found: {len(items)}")
        return JsonResponse({'success': True, 'items': items})

    except Exception as e:
        print(f"Error in api_letters_signed_urls: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


def api_digits(request):
    """Return JSON list of digits (0-9) with image URLs.

    Reads Firestore document OUD3xJakGcN5JgMNqpnn from Vocabulary collection.
    Returns: {success, items: [{id, name, url}, ...]}
    """
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)

    if firebase_admin is None:
        return JsonResponse({'success': False, 'error': 'firebase-admin not installed'}, status=500)

    try:
        # initialize firebase-admin if needed
        if not firebase_admin._apps:
            sa_path = os.path.join(os.path.dirname(__file__), '..', 'asl_web', 'firebase-service-account.json')
            if not os.path.exists(sa_path):
                return JsonResponse({'success': False, 'error': f'Service account not found: {sa_path}'}, status=500)
            cred = credentials.Certificate(sa_path)
            firebase_admin.initialize_app(cred)

        db = firestore.client(database_id='aslweb')
        items = []
        
        # Read specific document for digits
        doc_ref = db.collection('Vocabulary').document('OUD3xJakGcN5JgMNqpnn')
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict() or {}
            print(f"Digits document keys: {list(data.keys())}")
            # Iterate through all fields - handle both '0:' and '0' formats
            for key, val in data.items():
                if isinstance(val, str) and val.startswith('http'):
                    # Remove trailing colon if present
                    clean_key = key.rstrip(':')
                    items.append({'id': clean_key, 'name': clean_key, 'url': val})
                    print(f"Added digit: {clean_key} -> {val[:50]}...")
            
            # Sort by numeric value
            items.sort(key=lambda x: int(x['name']) if x['name'].isdigit() else 999)
        else:
            print("Digits document does not exist!")
        
        print(f"Total digits found: {len(items)}")
        return JsonResponse({'success': True, 'items': items})
    
    except Exception as e:
        print(f"Error in api_digits: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


def api_vocabulary_items(request):
    """Return JSON list of vocabulary items with multiple image URLs.

    Reads Firestore collection `Vocabulary` where documents contain arrays (0, 1, 2...) of image URLs.
    Returns: {success, items: [{id, images: [url1, url2, ...]}, ...]}
    """
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)

    if firebase_admin is None:
        return JsonResponse({'success': False, 'error': 'firebase-admin not installed'}, status=500)

    try:
        # initialize firebase-admin if needed
        if not firebase_admin._apps:
            sa_path = os.path.join(os.path.dirname(__file__), '..', 'asl_web', 'firebase-service-account.json')
            if not os.path.exists(sa_path):
                return JsonResponse({'success': False, 'error': f'Service account not found: {sa_path}'}, status=500)
            cred = credentials.Certificate(sa_path)
            firebase_admin.initialize_app(cred)

        db = firestore.client(database_id='aslweb')
        items = []
        coll = db.collection('Vocabulary')
        docs = coll.stream()
        
        for d in docs:
            data = d.to_dict() or {}
            # Document structure: numeric keys (0, 1, 2, ...) with string URLs
            image_urls = []
            # Collect all numeric-keyed fields or direct http URLs
            for key, val in data.items():
                if isinstance(val, str) and val.startswith('http'):
                    image_urls.append(val)
            
            # Also handle if data is stored as list (rare but possible)
            if not image_urls:
                # try extracting numeric keys in sorted order
                numeric_keys = sorted([k for k in data.keys() if k.isdigit()], key=int)
                for k in numeric_keys:
                    v = data[k]
                    if isinstance(v, str) and v.startswith('http'):
                        image_urls.append(v)
            
            if image_urls:
                items.append({'id': d.id, 'images': image_urls})
        
        return JsonResponse({'success': True, 'items': items})
    
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

def api_recognize(request):
    """API nhận diện ASL từ frame ảnh - TÍCH HỢP CHỮ SỐ VÀ CHỮ CÁI"""
    if request.method == 'POST':
        try:
            # Parse JSON body
            import json
            try:
                body = json.loads(request.body)
                image_data = body.get('image')
                recognition_mode = body.get('mode', 'all')  # 'all', 'letters', 'numbers'
            except:
                # Fallback to POST form data
                image_data = request.POST.get('image')
                recognition_mode = request.POST.get('mode', 'all')
            
            print(f"🎯 Recognition mode received: {recognition_mode}")
            
            # Load recognizers
            asl_recognizer = get_asl_recognizer(ASL_MODEL_PATH)
            digit_recognizer = get_digit_recognizer(DIGIT_MODEL_PATH)
            
            print(f"🔤 ASL Recognizer ready: {asl_recognizer is not None and asl_recognizer.is_initialized}")
            print(f"🔢 Digit Recognizer ready: {digit_recognizer is not None and digit_recognizer.is_initialized}")
            
            # Nhận frame ảnh từ frontend
            if not image_data:
                return JsonResponse({
                    'success': False,
                    'prediction': 'Không có dữ liệu ảnh',
                    'confidence': 0,
                    'hand_detected': False
                })
            
            # Decode base64 image
            try:
                if ';base64,' in image_data:
                    format, imgstr = image_data.split(';base64,')
                else:
                    imgstr = image_data
                
                image_bytes = base64.b64decode(imgstr)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    return JsonResponse({
                        'success': False,
                        'prediction': 'Cannot decode image',
                        'confidence': 0,
                        'hand_detected': False
                    })
                    
            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'prediction': f'Lỗi giải mã ảnh: {str(e)}',
                    'confidence': 0,
                    'hand_detected': False
                })
            
            # Quick MediaPipe hand presence check: nếu ASL recognizer có sẵn,
            # dùng MediaPipe để kiểm tra có tay trong frame hay không. Nếu không có
            # thì trả về ngay `--` / hand_detected=False để client không tiếp tục xử lý.
            asl_recognizer = get_asl_recognizer(ASL_MODEL_PATH)
            digit_recognizer = get_digit_recognizer(DIGIT_MODEL_PATH)

            if asl_recognizer and asl_recognizer.is_initialized:
                try:
                    rgb_for_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Use the MediaPipe hands detector directly for a lightweight check
                    results = asl_recognizer.hands.process(rgb_for_mp)
                    if not results or not results.multi_hand_landmarks:
                        # No hand detected - return quick response
                        return JsonResponse({
                            'success': True,
                            'prediction': '--',
                            'confidence': 0,
                            'hand_detected': False,
                            'type': 'character',
                            'mode': recognition_mode
                        })
                except Exception as e:
                    # If MediaPipe check fails, continue to regular pipeline
                    print(f"Warning: MediaPipe presence check failed: {e}")

            # Xử lý nhận diện theo mode - FIXED LOGIC
            prediction = "--"
            confidence = 0.0
            hand_detected = False
            
            if recognition_mode == 'numbers':
                # Chỉ sử dụng model chữ số
                if digit_recognizer and digit_recognizer.is_initialized:
                    try:
                        prediction, confidence = digit_recognizer.predict(frame)
                        prediction = str(prediction)
                        print(f"🔢 Digit prediction: {prediction}, confidence: {confidence}")
                        
                        # Kiểm tra kết quả hợp lệ
                        if prediction in "0123456789" and confidence > 0.1:
                            hand_detected = True
                        else:
                            prediction = "--"
                            confidence = 0.0
                            hand_detected = False
                    except Exception as e:
                        print(f"❌ Digit recognition error: {e}")
                        prediction = "--"
                        confidence = 0.0
                else:
                    return JsonResponse({
                        'success': False,
                        'prediction': 'Model chữ số chưa sẵn sàng',
                        'confidence': 0,
                        'hand_detected': False
                    })
                    
            elif recognition_mode == 'letters':
                # Sử dụng MediaPipe ROI + smoothing giống bản desktop
                if asl_recognizer and asl_recognizer.is_initialized:
                    try:
                        raw_pred, raw_conf, _bbox = asl_recognizer.process_frame(frame)
                        print(f"🔤 Letter (MP ROI) raw: {raw_pred}, conf: {raw_conf:.3f}")

                        if raw_pred and raw_pred.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and raw_conf > 0.1:
                            prediction = raw_pred.upper()
                            confidence = float(raw_conf)
                            hand_detected = getattr(asl_recognizer, 'last_hand_detected', True)
                        else:
                            # Fallback: dùng predict toàn khung như trước đây
                            fb_pred, fb_conf = asl_recognizer.predict(frame)
                            print(f"↩️ Fallback predict: {fb_pred}, conf: {fb_conf:.3f}")
                            if fb_pred and fb_pred.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and fb_conf > 0.1:
                                prediction = fb_pred.upper()
                                confidence = float(fb_conf)
                                hand_detected = True
                            else:
                                prediction = "--"
                                confidence = 0.0
                                hand_detected = getattr(asl_recognizer, 'last_hand_detected', False)
                    except Exception as e:
                        print(f"❌ Letter recognition error: {e}")
                        prediction = "--"
                        confidence = 0.0
                else:
                    return JsonResponse({
                        'success': False,
                        'prediction': 'Model chữ cái chưa sẵn sàng',
                        'confidence': 0,
                        'hand_detected': False
                    })
                    
            else:  # mode = 'all'
                # Thử cả hai model, ưu tiên model có confidence cao hơn
                best_prediction = "--"
                best_confidence = 0.0
                
                # Thử ASL recognizer (chữ cái) với MediaPipe ROI
                if asl_recognizer and asl_recognizer.is_initialized:
                    try:
                        asl_pred_raw, asl_conf_raw, _bbox = asl_recognizer.process_frame(frame)
                        asl_hand = getattr(asl_recognizer, 'last_hand_detected', False)
                        cand_pred = None
                        cand_conf = 0.0
                        cand_hand = asl_hand
                        if asl_pred_raw and asl_pred_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and asl_conf_raw > 0.1:
                            cand_pred = asl_pred_raw.upper()
                            cand_conf = float(asl_conf_raw)
                        else:
                            # Fallback to legacy predict
                            fb_pred, fb_conf = asl_recognizer.predict(frame)
                            if fb_pred and fb_pred.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and fb_conf > 0.1:
                                cand_pred = fb_pred.upper()
                                cand_conf = float(fb_conf)
                                cand_hand = True
                        if cand_pred and cand_conf > best_confidence:
                            best_prediction = cand_pred
                            best_confidence = cand_conf
                            hand_detected = cand_hand
                    except Exception as e:
                        print(f"❌ ASL recognition error: {e}")
                
                # Thử digit recognizer (chữ số)
                if digit_recognizer and digit_recognizer.is_initialized:
                    try:
                        digit_pred, digit_conf = digit_recognizer.predict(frame)
                        if digit_conf > best_confidence and digit_pred in "0123456789" and digit_conf > 0.1:
                            best_prediction = digit_pred
                            best_confidence = digit_conf
                            hand_detected = True
                    except Exception as e:
                        print(f"❌ Digit recognition error: {e}")
                
                prediction = best_prediction
                confidence = best_confidence
            
            print(f"🎯 Final Result: {prediction}, Confidence: {confidence:.2f}, Hand: {hand_detected}")
            
            return JsonResponse({
                'success': True,
                'prediction': prediction,
                'confidence': float(confidence * 100),  # Convert to percentage
                'type': 'character',
                'mode': recognition_mode,
                'hand_detected': hand_detected
            })
            
        except Exception as e:
            print(f"❌ API recognition error: {e}")
            traceback.print_exc()
            return JsonResponse({
                'success': False,
                'prediction': f'Lỗi hệ thống: {str(e)}',
                'confidence': 0,
                'hand_detected': False
            })
    
    return JsonResponse({
        'success': False, 
        'prediction': 'Method not allowed', 
        'confidence': 0,
        'hand_detected': False
    })

def api_words_list(request):
    """API lấy danh sách từ vựng + video từ Firebase.
    
    Cấu trúc Firebase: Collection 'Vocabulary' > Document > Fields: {WORD_NAME: video_url}
    Returns: {success, items: [{word, video_url}, ...]}
    """
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)

    if firebase_admin is None:
        return JsonResponse({'success': False, 'error': 'firebase-admin not installed'}, status=500)

    try:
        # Initialize firebase-admin if needed
        if not firebase_admin._apps:
            sa_path = os.path.join(os.path.dirname(__file__), '..', 'asl_web', 'firebase-service-account.json')
            if not os.path.exists(sa_path):
                return JsonResponse({'success': False, 'error': f'Service account not found: {sa_path}'}, status=500)
            cred = credentials.Certificate(sa_path)
            firebase_admin.initialize_app(cred)

        db = firestore.client(database_id='aslweb')
        items = []
        
        # Read specific document from Vocabulary collection
        # Document ID: JCr5Z2sA8lg6P17OOqIp
        # Fields: {WORD_NAME: video_url}
        doc_ref = db.collection('Vocabulary').document('JCr5Z2sA8lg6P17OOqIp')
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict() or {}
            # Each field in the document is a word with its video URL
            for word_name, video_url in data.items():
                if isinstance(video_url, str) and video_url.startswith('http'):
                    items.append({
                        'id': f"{doc.id}_{word_name}",
                        'word': word_name,
                        'video_url': video_url
                    })
        else:
            print("❌ Document JCr5Z2sA8lg6P17OOqIp not found")
        
        # Sort alphabetically
        items.sort(key=lambda x: x['word'].lower())
        
        print(f"📚 Total words found from Vocabulary: {len(items)}")
        for item in items[:5]:
            print(f"  - {item['word']}: {item['video_url'][:50]}...")
        
        return JsonResponse({'success': True, 'items': items})
    
    except Exception as e:
        print(f"Error in api_words_list: {e}")
        traceback.print_exc()
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


def api_recognize_words(request):
    """API nhận diện từ vựng ASL với model thật - HOÀN CHỈNH"""
    if request.method == 'POST':
        start_time = time.time()
        
        try:
            print("\n" + "="*60)
            print("🔄 API WORD RECOGNITION CALLED")
            print("="*60)
            
            # ==================== XỬ LÝ RESET REQUEST ====================
            if request.POST.get('reset') == 'true':
                print("🔄 Reset request received")
                recognizer = get_word_recognizer_lazy(WORD_MODEL_PATH)
                if recognizer:
                    recognizer.reset()
                    print("✅ Reset successful")
                    return JsonResponse({
                        'success': True,
                        'message': 'Reset successful',
                        'reset_time': time.time()
                    })
                else:
                    print("❌ Recognizer not available for reset")
                    return JsonResponse({
                        'success': False,
                        'message': 'Recognizer not available',
                        'reset_time': time.time()
                    })
            
            # ==================== KIỂM TRA DỮ LIỆU ẢNH ====================
            image_data = request.POST.get('image')
            if not image_data:
                print("❌ No image data received")
                return JsonResponse({
                    'success': False,
                    'prediction': 'Không có dữ liệu ảnh',
                    'confidence': 0,
                    'processing_time': 0,
                    'error_type': 'NO_IMAGE_DATA'
                })
            
            print(f"📷 Image data received - Length: {len(image_data)}")
            
            # ==================== DECODE BASE64 IMAGE ====================
            decode_start = time.time()
            try:
                # Xử lý data URL format
                if ';base64,' in image_data:
                    format, imgstr = image_data.split(';base64,')
                    print(f"📁 Image format: {format}")
                else:
                    imgstr = image_data
                    print("📁 Raw base64 data")
                
                # Decode base64
                image_bytes = base64.b64decode(imgstr)
                print(f"🔧 Decoded bytes: {len(image_bytes)}")
                
                # Decode image với OpenCV
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print("❌ Cannot decode image - invalid format")
                    return JsonResponse({
                        'success': False,
                        'prediction': 'Không thể giải mã ảnh',
                        'confidence': 0,
                        'processing_time': 0,
                        'error_type': 'IMAGE_DECODE_FAILED'
                    })
                
                decode_time = time.time() - decode_start
                print(f"✅ Image decoded - Shape: {frame.shape}, Time: {decode_time:.3f}s")
                
            except Exception as e:
                print(f"❌ Image decoding error: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'prediction': f'Lỗi giải mã ảnh: {str(e)}',
                    'confidence': 0,
                    'processing_time': 0,
                    'error_type': 'DECODING_ERROR'
                })
            
            # ==================== LOAD MODEL RECOGNIZER ====================
            model_load_start = time.time()
            try:
                recognizer = get_word_recognizer_lazy(WORD_MODEL_PATH)
                
                if not recognizer:
                    print("❌ Word recognition model not ready")
                    return JsonResponse({
                        'success': False,
                        'prediction': 'Model nhận diện chưa sẵn sàng',
                        'confidence': 0,
                        'processing_time': 0,
                        'error_type': 'MODEL_NOT_READY'
                    })
                
                model_load_time = time.time() - model_load_start
                print(f"✅ Model loaded - Time: {model_load_time:.3f}s")
                print(f"🔧 Model initialized: {recognizer.is_initialized}")
                print(f"📊 Buffer size: {len(recognizer.buffer)}")
                
            except Exception as e:
                print(f"❌ Model loading error: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'prediction': f'Lỗi tải model: {str(e)}',
                    'confidence': 0,
                    'processing_time': 0,
                    'error_type': 'MODEL_LOAD_ERROR'
                })
            
            # ==================== NHẬN DIỆN TỪ VỰNG ====================
            recognition_start = time.time()
            try:
                print("🎯 Starting word recognition...")
                
                # Xử lý frame và nhận diện
                prediction, confidence = recognizer.process_frame(frame)
                
                recognition_time = time.time() - recognition_start
                total_time = time.time() - start_time
                
                print(f"✅ Recognition completed")
                print(f"📊 Result: {prediction} (Confidence: {confidence:.1f}%)")
                print(f"⏱️ Times - Decode: {decode_time:.3f}s, Model: {model_load_time:.3f}s, Recognition: {recognition_time:.3f}s, Total: {total_time:.3f}s")
                print(f"📈 Buffer status: {len(recognizer.buffer)} frames")
                
                # Phân loại kết quả
                result_type = "SUCCESS"
                if "lỗi" in prediction.lower() or "error" in prediction.lower():
                    result_type = "ERROR"
                elif "đang thu thập" in prediction.lower() or "collecting" in prediction.lower():
                    result_type = "COLLECTING"
                elif "không phát hiện" in prediction.lower() or "no hand" in prediction.lower():
                    result_type = "NO_HAND"
                elif confidence < 30:
                    result_type = "LOW_CONFIDENCE"
                
                # Chuẩn bị response
                response_data = {
                    'success': True,
                    'prediction': prediction,
                    'confidence': float(confidence),
                    'type': 'word',
                    'mode': 'real_model',
                    'processing_time': total_time,
                    'breakdown': {
                        'image_decode': decode_time,
                        'model_load': model_load_time,
                        'recognition': recognition_time
                    },
                    'buffer_status': {
                        'current_size': len(recognizer.buffer),
                        'required_size': 15,
                        'progress_percent': min(100, (len(recognizer.buffer) / 15) * 100)
                    },
                    'result_type': result_type,
                    'timestamp': time.time()
                }
                
                # Thêm debug info nếu đang thu thập dữ liệu
                if result_type == "COLLECTING":
                    response_data['collection_progress'] = {
                        'current_frames': len(recognizer.buffer),
                        'required_frames': 15,
                        'progress_percent': min(100, (len(recognizer.buffer) / 15) * 100)
                    }
                
                print(f"📤 Sending response - Type: {result_type}")
                return JsonResponse(response_data)
                
            except Exception as e:
                recognition_time = time.time() - recognition_start
                total_time = time.time() - start_time
                
                print(f"❌ Recognition error: {str(e)}")
                traceback.print_exc()
                
                return JsonResponse({
                    'success': False,
                    'prediction': f'Lỗi nhận diện: {str(e)}',
                    'confidence': 0,
                    'processing_time': total_time,
                    'error_type': 'RECOGNITION_ERROR',
                    'breakdown': {
                        'image_decode': decode_time,
                        'model_load': model_load_time,
                        'recognition': recognition_time
                    }
                })
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"💥 General API error: {str(e)}")
            traceback.print_exc()
            
            return JsonResponse({
                'success': False,
                'prediction': f'Lỗi hệ thống: {str(e)}',
                'confidence': 0,
                'processing_time': total_time,
                'error_type': 'GENERAL_ERROR'
            })
    
    # ==================== METHOD NOT ALLOWED ====================
    print("❌ Method not allowed - GET request received")
    return JsonResponse({
        'success': False,
        'prediction': 'Method not allowed',
        'confidence': 0,
        'processing_time': 0,
        'error_type': 'METHOD_NOT_ALLOWED'
    })


# ==================== API RECOGNIZE WORDS V2 - YOLO + Transformer ====================
def api_recognize_words_v2(request):
    """API nhận diện từ vựng ASL V2 với YOLO + Transformer model"""
    if request.method == 'POST':
        start_time = time.time()
        
        try:
            print("\n" + "="*60)
            print("🔄 API WORD RECOGNITION V2 CALLED (YOLO + Transformer)")
            print("="*60)
            
            # ==================== XỬ LÝ RESET REQUEST ====================
            if request.POST.get('reset') == 'true':
                print("🔄 Reset request received")
                from .word_recognizer_v3 import get_word_recognizer_v3
                recognizer = get_word_recognizer_v3()
                if recognizer:
                    recognizer.reset()
                    print("✅ Reset successful")
                    return JsonResponse({
                        'success': True,
                        'message': 'Reset successful',
                        'reset_time': time.time()
                    })
                else:
                    print("❌ Recognizer V3 not available for reset")
                    return JsonResponse({
                        'success': False,
                        'message': 'Recognizer not available',
                        'reset_time': time.time()
                    })
            
            # ==================== KIỂM TRA DỮ LIỆU ẢNH ====================
            image_data = request.POST.get('image')
            if not image_data:
                print("❌ No image data received")
                return JsonResponse({
                    'success': False,
                    'prediction': 'Không có dữ liệu ảnh',
                    'confidence': 0,
                    'processing_time': 0,
                    'error_type': 'NO_IMAGE_DATA'
                })
            
            print(f"📷 Image data received - Length: {len(image_data)}")
            
            # ==================== DECODE BASE64 IMAGE ====================
            decode_start = time.time()
            try:
                # Xử lý data URL format
                if ';base64,' in image_data:
                    format, imgstr = image_data.split(';base64,')
                    print(f"📁 Image format: {format}")
                else:
                    imgstr = image_data
                    print("📁 Raw base64 data")
                
                # Decode base64
                image_bytes = base64.b64decode(imgstr)
                print(f"🔧 Decoded bytes: {len(image_bytes)}")
                
                # Decode image với OpenCV
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print("❌ Cannot decode image - invalid format")
                    return JsonResponse({
                        'success': False,
                        'prediction': 'Không thể giải mã ảnh',
                        'confidence': 0,
                        'processing_time': 0,
                        'error_type': 'IMAGE_DECODE_FAILED'
                    })
                
                decode_time = time.time() - decode_start
                print(f"✅ Image decoded - Shape: {frame.shape}, Time: {decode_time:.3f}s")
                
            except Exception as e:
                print(f"❌ Image decoding error: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'prediction': f'Lỗi giải mã ảnh: {str(e)}',
                    'confidence': 0,
                    'processing_time': time.time() - start_time,
                    'error_type': 'DECODE_ERROR'
                })
            
            # ==================== LOAD RECOGNIZER V3 ====================
            model_load_start = time.time()
            try:
                from .word_recognizer_v3 import get_word_recognizer_v3, init_word_recognizer_v3
                
                recognizer = get_word_recognizer_v3()
                if recognizer is None:
                    print("🚀 Initializing WordRecognizerV3...")
                    success = init_word_recognizer_v3()
                    if not success:
                        return JsonResponse({
                            'success': False,
                            'prediction': 'Không thể khởi tạo model V3',
                            'confidence': 0,
                            'processing_time': time.time() - start_time,
                            'error_type': 'MODEL_INIT_FAILED'
                        })
                    recognizer = get_word_recognizer_v3()
                
                model_load_time = time.time() - model_load_start
                print(f"✅ Recognizer V3 ready - Time: {model_load_time:.3f}s")
                
            except Exception as e:
                print(f"❌ Model load error: {str(e)}")
                traceback.print_exc()
                return JsonResponse({
                    'success': False,
                    'prediction': f'Lỗi tải model: {str(e)}',
                    'confidence': 0,
                    'processing_time': time.time() - start_time,
                    'error_type': 'MODEL_LOAD_ERROR'
                })
            
            # ==================== RECOGNITION ====================
            recognition_start = time.time()
            try:
                prediction, confidence = recognizer.process_frame(frame)
                
                recognition_time = time.time() - recognition_start
                total_time = time.time() - start_time
                
                print(f"🎯 Prediction: {prediction} ({confidence:.1f}%)")
                print(f"⏱️ Total time: {total_time:.3f}s")
                
                # Determine result type
                if "thu thập" in prediction.lower() or "đang" in prediction.lower():
                    result_type = "COLLECTING"
                elif confidence < 30:
                    result_type = "LOW_CONFIDENCE"
                else:
                    result_type = "PREDICTION"
                
                response_data = {
                    'success': True,
                    'prediction': prediction,
                    'confidence': round(confidence, 1),
                    'processing_time': round(total_time * 1000, 1),  # ms
                    'breakdown': {
                        'image_decode': round(decode_time * 1000, 1),
                        'model_load': round(model_load_time * 1000, 1),
                        'recognition': round(recognition_time * 1000, 1)
                    },
                    'buffer_status': {
                        'current_size': len(recognizer.buffer.feature_buffer) if recognizer.buffer else 0,
                        'required_size': 15,
                        'progress_percent': min(100, (len(recognizer.buffer.feature_buffer) / 15) * 100) if recognizer.buffer else 0
                    },
                    'result_type': result_type,
                    'timestamp': time.time()
                }
                
                # Thêm collection progress nếu đang thu thập
                if result_type == "COLLECTING" and recognizer.buffer:
                    response_data['collection_progress'] = {
                        'current_frames': len(recognizer.buffer.feature_buffer),
                        'required_frames': 15,
                        'progress_percent': min(100, (len(recognizer.buffer.feature_buffer) / 15) * 100)
                    }
                
                print(f"📤 Sending response - Type: {result_type}")
                return JsonResponse(response_data)
                
            except Exception as e:
                recognition_time = time.time() - recognition_start
                total_time = time.time() - start_time
                
                print(f"❌ Recognition error: {str(e)}")
                traceback.print_exc()
                
                return JsonResponse({
                    'success': False,
                    'prediction': f'Lỗi nhận diện: {str(e)}',
                    'confidence': 0,
                    'processing_time': total_time,
                    'error_type': 'RECOGNITION_ERROR',
                    'breakdown': {
                        'image_decode': decode_time,
                        'model_load': model_load_time,
                        'recognition': recognition_time
                    }
                })
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"💥 General API error: {str(e)}")
            traceback.print_exc()
            
            return JsonResponse({
                'success': False,
                'prediction': f'Lỗi hệ thống: {str(e)}',
                'confidence': 0,
                'processing_time': total_time,
                'error_type': 'GENERAL_ERROR'
            })
    
    # ==================== METHOD NOT ALLOWED ====================
    print("❌ Method not allowed - GET request received")
    return JsonResponse({
        'success': False,
        'prediction': 'Method not allowed',
        'confidence': 0,
        'processing_time': 0,
        'error_type': 'METHOD_NOT_ALLOWED'
    })