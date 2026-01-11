// learning/static/learning/js/camera_real.js
class ASLRealRecognizer {
    constructor() {
        // Khởi tạo các element cơ bản
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.captureCanvas = document.getElementById('capture-canvas');
        
        // Khởi tạo context với kiểm tra tồn tại
        if (this.canvas) {
            this.ctx = this.canvas.getContext('2d');
        } else {
            console.error('❌ Canvas element not found');
            this.ctx = null;
        }
        
        if (this.captureCanvas) {
            this.captureCtx = this.captureCanvas.getContext('2d');
        } else {
            console.error('❌ Capture canvas element not found');
            this.captureCtx = null;
        }
        
        // Control buttons
        this.startBtn = document.getElementById('start-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.captureBtn = document.getElementById('capture-btn');
        this.recognizeCaptureBtn = document.getElementById('recognize-capture-btn');
        this.retakeBtn = document.getElementById('retake-btn');
        this.saveCaptureBtn = document.getElementById('save-capture-btn');
        this.speakBtn = document.getElementById('speak-btn');
        this.toggleSpeechBtn = document.getElementById('toggle-speech-btn');
        this.testSpeechBtn = document.getElementById('test-speech-btn');
        
        // Real-time results
        this.result = document.getElementById('prediction-result');
        this.confidence = document.getElementById('confidence');
        this.confidenceBar = document.getElementById('confidence-bar');
        this.status = document.getElementById('status');
        
        // Capture results
        this.captureResult = document.getElementById('capture-prediction');
        this.captureConfidence = document.getElementById('capture-confidence');
        this.captureConfidenceBar = document.getElementById('capture-confidence-bar');
        this.captureStatus = document.getElementById('capture-status');
        this.capturedImage = document.getElementById('captured-image');
        this.captureSection = document.getElementById('capture-section');
        
        this.speechStatus = document.getElementById('speech-status');
        this.modeDisplay = document.getElementById('mode-display');
        this.modelDisplay = document.getElementById('model-display');
        this.currentModeInfo = document.getElementById('current-mode-info');
        
        // Mode buttons
        this.modeAll = document.getElementById('mode-all');
        this.modeLetters = document.getElementById('mode-letters');
        this.modeNumbers = document.getElementById('mode-numbers');
        
        this.stream = null;
        this.isRunning = false;
        this.autoSpeech = false;
        this.lastPrediction = '';
        this.recognitionInterval = null;
        this.currentMode = 'letters'; // 'all', 'letters', 'numbers' - Default to letters
        this.capturedImageData = null;
        
        this.speechSynth = window.speechSynthesis;
        this.voices = [];
        
        // Thiết lập kích thước canvas
        if (this.canvas) {
            this.canvas.width = 640;
            this.canvas.height = 480;
        }
        
        if (this.captureCanvas) {
            this.captureCanvas.width = 640;
            this.captureCanvas.height = 480;
        }
        
        this.initializeEventListeners();
        this.loadVoices();
        this.updateModeDisplay();
        
        console.log('✅ ASLRealRecognizer initialized');
    }
    
    initializeEventListeners() {
        // Camera controls
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        
        // Capture controls
        if (this.captureBtn) {
            this.captureBtn.addEventListener('click', () => this.captureImage());
        }
        if (this.recognizeCaptureBtn) {
            this.recognizeCaptureBtn.addEventListener('click', () => this.recognizeCapturedImage());
        }
        if (this.retakeBtn) {
            this.retakeBtn.addEventListener('click', () => this.retakeImage());
        }
        if (this.saveCaptureBtn) {
            this.saveCaptureBtn.addEventListener('click', () => this.saveCapturedImage());
        }
        
        // Speech controls
        this.speakBtn.addEventListener('click', () => this.speakText());
        this.toggleSpeechBtn.addEventListener('click', () => this.toggleAutoSpeech());
        this.testSpeechBtn.addEventListener('click', () => this.testSpeech());
        
        // Mode selection events
        this.modeAll.addEventListener('click', () => this.setMode('all'));
        this.modeLetters.addEventListener('click', () => this.setMode('letters'));
        this.modeNumbers.addEventListener('click', () => this.setMode('numbers'));
        
        // Upload controls
        this.initializeUploadControls();
    }
    
    initializeUploadControls() {
        const uploadInput = document.getElementById('upload-input');
        const browseBtn = document.getElementById('browse-btn');
        const uploadArea = document.getElementById('upload-area');
        const recognizeUploadBtn = document.getElementById('recognize-upload-btn');
        const clearUploadBtn = document.getElementById('clear-upload-btn');
        
        if (!uploadInput || !browseBtn || !uploadArea) return;
        
        // Browse button click
        browseBtn.addEventListener('click', () => uploadInput.click());
        
        // File input change
        uploadInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files[0]));
        
        // Drag & drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                this.handleFileSelect(file);
            }
        });
        
        // Recognize uploaded image
        if (recognizeUploadBtn) {
            recognizeUploadBtn.addEventListener('click', () => this.recognizeUploadedImage());
        }
        
        // Clear upload
        if (clearUploadBtn) {
            clearUploadBtn.addEventListener('click', () => this.clearUpload());
        }
        
        // Save uploaded image
        const saveUploadBtn = document.getElementById('save-upload-btn');
        if (saveUploadBtn) {
            saveUploadBtn.addEventListener('click', () => this.saveUploadedImage());
        }
    }
    
    handleFileSelect(file) {
        if (!file) return;
        
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.updateUploadStatus('❌ Vui lòng chọn file ảnh', 'danger');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            this.uploadedImageData = e.target.result;
            this.showUploadPreview(e.target.result);
            this.updateUploadStatus('✅ Đã tải ảnh thành công', 'success');
        };
        reader.onerror = () => {
            this.updateUploadStatus('❌ Lỗi khi đọc file', 'danger');
        };
        reader.readAsDataURL(file);
    }
    
    showUploadPreview(imageData) {
        const placeholder = document.getElementById('upload-placeholder');
        const preview = document.getElementById('upload-preview');
        const uploadImage = document.getElementById('upload-image');
        
        if (placeholder) placeholder.style.display = 'none';
        if (preview) preview.style.display = 'block';
        if (uploadImage) uploadImage.src = imageData;
    }
    
    async recognizeUploadedImage() {
        if (!this.uploadedImageData) {
            this.updateUploadStatus('❌ Chưa có ảnh để nhận diện', 'danger');
            return;
        }
        
        try {
            this.updateUploadStatus('🔍 Đang nhận diện ảnh...', 'warning');
            this.updateCaptureStatus('ĐANG NHẬN DIỆN...');
            
            const response = await this.sendToServer(this.uploadedImageData);
            
            if (response.success) {
                // Update capture results section (unified display)
                this.updateCaptureResult(response.prediction, response.confidence);
                this.updateCaptureStatus('HOÀN TẤT');
                this.updateImageSource('Ảnh tải lên');
                
                this.updateUploadStatus('✅ Nhận diện thành công', 'success');
                this.updateStatus(`Kết quả từ ảnh tải lên: ${response.prediction}`);
                
                // Auto speak if enabled
                if (this.autoSpeech && response.prediction !== '--' && !response.prediction.includes('Lỗi')) {
                    this.speakTextContent(response.prediction);
                }
            } else {
                this.updateCaptureStatus('LỖI NHẬN DIỆN');
                this.updateUploadStatus('❌ Lỗi nhận diện', 'danger');
            }
        } catch (error) {
            console.error('❌ Lỗi nhận diện ảnh tải lên:', error);
            this.updateCaptureStatus('LỖI KẾT NỐI');
            this.updateUploadStatus('❌ Lỗi kết nối server', 'danger');
        }
    }
    
    clearUpload() {
        const uploadInput = document.getElementById('upload-input');
        const placeholder = document.getElementById('upload-placeholder');
        const preview = document.getElementById('upload-preview');
        
        if (uploadInput) uploadInput.value = '';
        if (placeholder) placeholder.style.display = 'block';
        if (preview) preview.style.display = 'none';
        
        this.uploadedImageData = null;
        
        // Reset capture results
        this.updateCaptureResult('--', 0);
        this.updateCaptureStatus('CHƯA CÓ ẢNH');
        this.updateImageSource('Chưa xác định');
        
        this.updateUploadStatus('🔄 Đã xóa ảnh. Sẵn sàng tải ảnh mới', 'info');
    }
    
    updateUploadStatus(message, type) {
        const uploadStatus = document.getElementById('upload-status');
        if (uploadStatus) {
            uploadStatus.innerHTML = `<i class="fas fa-${type === 'success' ? 'check-circle' : type === 'danger' ? 'exclamation-circle' : 'info-circle'} me-2"></i>${message}`;
            uploadStatus.className = `fs-6 text-${type} mt-2`;
        }
    }
    
    updateImageSource(source) {
        const imageSource = document.getElementById('image-source');
        if (imageSource) {
            imageSource.textContent = source;
        }
    }
    
    saveUploadedImage() {
        if (!this.uploadedImageData) {
            this.updateUploadStatus('❌ Không có ảnh để lưu', 'danger');
            return;
        }
        
        const link = document.createElement('a');
        link.download = `asl-upload-${Date.now()}.png`;
        link.href = this.uploadedImageData;
        link.click();
        
        this.updateUploadStatus('✅ Đã lưu ảnh thành công', 'success');
    }
    
    setMode(mode) {
        this.currentMode = mode;
        
        // Update button states
        document.querySelectorAll('.recognition-mode').forEach(btn => {
            btn.classList.remove('active', 'btn-primary');
            btn.classList.add('btn-outline-primary');
        });
        
        const activeBtn = document.getElementById(`mode-${mode}`);
        activeBtn.classList.add('active', 'btn-primary');
        activeBtn.classList.remove('btn-outline-primary');
        
        this.updateModeDisplay();
        
        console.log(`🎯 Mode changed to: ${mode}`);
        this.updateStatus(`Đã chuyển sang chế độ: ${this.getModeDisplayName(mode)}`);
        
        // Reset prediction khi đổi mode
        if (this.isRunning) {
            this.updateResult('--', 0);
        }
    }
    
    getModeDisplayName(mode) {
        switch(mode) {
            case 'all': return 'Tất cả (A-Z + 0-9)';
            case 'letters': return 'Chỉ chữ cái (A-Z)';
            case 'numbers': return 'Chỉ chữ số (0-9)';
            default: return 'Tất cả';
        }
    }
    
    updateModeDisplay() {
        const displayNames = {
            'all': 'Tất cả',
            'letters': 'Chỉ chữ cái', 
            'numbers': 'Chỉ chữ số'
        };
        
        const modelNames = {
            'all': 'Chữ cái + Chữ số',
            'letters': 'Chữ cái',
            'numbers': 'Chữ số'
        };
        
        if (this.modeDisplay) {
            this.modeDisplay.textContent = displayNames[this.currentMode];
        }
        if (this.modelDisplay) {
            this.modelDisplay.textContent = modelNames[this.currentMode];
        }
        if (this.currentModeInfo) {
            this.currentModeInfo.textContent = `Đang nhận diện: ${this.getModeDisplayName(this.currentMode)}`;
        }
    }
    
    loadVoices() {
        this.voices = this.speechSynth.getVoices();
        if (this.voices.length === 0) {
            this.speechSynth.onvoiceschanged = () => {
                this.voices = this.speechSynth.getVoices();
                console.log('Voices loaded:', this.voices.length);
            };
        }
    }
    
    async startCamera() {
        try {
            console.log('🚀 Starting camera and AI recognition...');
            this.updateStatus('Đang khởi động camera...');
            
            // Kiểm tra hỗ trợ camera
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Trình duyệt không hỗ trợ truy cập camera');
            }

            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user',
                    frameRate: { ideal: 30 }
                } 
            });
            
            this.video.srcObject = this.stream;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            if (this.captureBtn) this.captureBtn.disabled = false;
            this.speakBtn.disabled = false;
            this.isRunning = true;
            
            console.log('📹 Video element setup, waiting for metadata...');
            
            this.video.onloadedmetadata = () => {
                console.log('✅ Camera metadata loaded');
                this.video.play().then(() => {
                    console.log('▶️ Video playing successfully');
                    this.updateStatus('Camera đã sẵn sàng. Đang nhận diện...');
                    this.startRecognition();
                }).catch(error => {
                    console.error('❌ Lỗi phát video:', error);
                    this.updateStatus('Lỗi: Không thể phát video');
                });
            };
            
            // Fallback nếu onloadedmetadata không fire
            setTimeout(() => {
                if (this.video.readyState >= 2 && !this.recognitionInterval) {
                    console.log('⚠️ Metadata event missed, starting recognition via timeout');
                    this.video.play();
                    this.startRecognition();
                }
            }, 2000);
            
        } catch (error) {
            console.error('❌ Lỗi khi truy cập camera:', error);
            this.updateStatus('Lỗi: Không thể truy cập camera');
            
            let errorMessage = 'Không thể truy cập camera. ';
            if (error.name === 'NotAllowedError') {
                errorMessage += 'Vui lòng cấp quyền truy cập camera.';
            } else if (error.name === 'NotFoundError') {
                errorMessage += 'Không tìm thấy camera.';
            } else if (error.name === 'NotSupportedError') {
                errorMessage += 'Trình duyệt không hỗ trợ camera.';
            } else {
                errorMessage += 'Lỗi không xác định.';
            }
            
            alert(errorMessage);
        }
    }
    
    startRecognition() {
        console.log('🎯 Starting real-time AI recognition...');
        this.updateStatus('AI đang nhận diện...');
        
        this.recognitionInterval = setInterval(() => {
            console.log(`⏱️ Interval tick - isRunning: ${this.isRunning}, readyState: ${this.video.readyState}`);
            if (this.isRunning && this.video.readyState === this.video.HAVE_ENOUGH_DATA) {
                console.log('✅ Conditions met, calling captureAndRecognize()');
                this.captureAndRecognize();
            } else {
                console.log('❌ Conditions not met for recognition');
            }
        }, 500);
    }
    
    async captureAndRecognize() {
        console.log('🎥 captureAndRecognize() called');
        try {
            // Kiểm tra context
            if (!this.ctx) {
                console.error('❌ Canvas context not available');
                return;
            }

            // Vẽ video lên canvas với flip horizontal để khớp với video
            this.ctx.save();
            this.ctx.scale(-1, 1);
            this.ctx.drawImage(this.video, -this.canvas.width, 0, this.canvas.width, this.canvas.height);
            this.ctx.restore();
            
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
            console.log('📸 Image captured, size:', imageData.length);
            
            // Gửi mode cùng với request
            const response = await this.sendToServer(imageData);
            
            if (response.success) {
                this.updateResult(response.prediction, response.confidence, response.hand_detected);
                
                // Tự động phát âm nếu enabled và có phát hiện tay
                if (response.hand_detected && this.autoSpeech && response.confidence > 70 && 
                    response.prediction !== this.lastPrediction &&
                    this.isValidPrediction(response.prediction)) {
                    this.speakText();
                }
                
                this.lastPrediction = response.prediction;
            } else {
                this.updateResult(response.prediction || 'Lỗi nhận diện', 0);
            }
            
        } catch (error) {
            console.error('❌ Lỗi trong captureAndRecognize:', error);
            this.updateResult('Lỗi xử lý', 0);
        }
    }
    
    // Chụp ảnh từ camera
    captureImage() {
        try {
            if (!this.captureCtx) {
                console.error('❌ Capture canvas context not available');
                this.updateCaptureStatus('❌ Lỗi hệ thống', 'danger');
                return;
            }

            // Vẽ frame hiện tại lên capture canvas
            this.captureCtx.save();
            this.captureCtx.scale(-1, 1);
            this.captureCtx.drawImage(this.video, -this.captureCanvas.width, 0, this.captureCanvas.width, this.captureCanvas.height);
            this.captureCtx.restore();
            
            // Lưu dữ liệu ảnh
            this.capturedImageData = this.captureCanvas.toDataURL('image/jpeg', 0.9);
            
            // Hiển thị ảnh đã chụp
            if (this.capturedImage) {
                this.capturedImage.src = this.capturedImageData;
                this.capturedImage.style.display = 'block';
            }
            
            if (this.captureSection) {
                this.captureSection.style.display = 'block';
            }
            
            // Cập nhật trạng thái
            this.updateCaptureStatus('✅ Đã chụp ảnh thành công', 'success');
            this.updateStatus('Đã chụp ảnh. Nhấn "NHẬN DIỆN ẢNH" để phân tích.');
            
            console.log('📸 Image captured successfully');
            
        } catch (error) {
            console.error('❌ Lỗi khi chụp ảnh:', error);
            this.updateCaptureStatus('❌ Lỗi khi chụp ảnh', 'danger');
        }
    }
    
    // Nhận diện ảnh đã chụp
    async recognizeCapturedImage() {
        if (!this.capturedImageData) {
            this.updateCaptureStatus('❌ Chưa có ảnh để nhận diện', 'danger');
            return;
        }
        
        try {
            this.updateCaptureStatus('🔍 Đang nhận diện ảnh...', 'warning');
            
            const response = await this.sendToServer(this.capturedImageData);
            
            if (response.success) {
                this.updateCaptureResult(response.prediction, response.confidence);
                this.updateCaptureStatus('✅ Nhận diện ảnh thành công', 'success');
                this.updateImageSource('Ảnh chụp từ camera');
                
                // Tự động phát âm kết quả (chỉ khi bật)
                if (this.autoSpeech && response.prediction !== '--' && !response.prediction.includes('Lỗi')) {
                    this.speakCaptureText(response.prediction);
                }
                
            } else {
                this.updateCaptureResult('Lỗi nhận diện', 0);
                this.updateCaptureStatus('❌ Lỗi nhận diện ảnh', 'danger');
            }
            
        } catch (error) {
            console.error('❌ Lỗi nhận diện ảnh:', error);
            this.updateCaptureResult('Lỗi kết nối', 0);
            this.updateCaptureStatus('❌ Lỗi kết nối server', 'danger');
        }
    }
    
    // Chụp lại ảnh
    retakeImage() {
        this.capturedImageData = null;
        if (this.capturedImage) {
            this.capturedImage.style.display = 'none';
        }
        if (this.captureSection) {
            this.captureSection.style.display = 'none';
        }
        this.updateCaptureResult('--', 0);
        this.updateCaptureStatus('🔄 Sẵn sàng chụp ảnh mới', 'info');
        this.updateStatus('Đã xóa ảnh. Sẵn sàng chụp ảnh mới.');
    }
    
    // Lưu ảnh đã chụp
    saveCapturedImage() {
        if (!this.capturedImageData) {
            alert('Chưa có ảnh để lưu!');
            return;
        }
        
        try {
            const link = document.createElement('a');
            link.download = `asl_capture_${new Date().getTime()}.jpg`;
            link.href = this.capturedImageData;
            link.click();
            
            this.updateCaptureStatus('💾 Đã lưu ảnh thành công', 'success');
            this.updateStatus('Đã lưu ảnh xuống thiết bị.');
            
        } catch (error) {
            console.error('❌ Lỗi khi lưu ảnh:', error);
            this.updateCaptureStatus('❌ Lỗi khi lưu ảnh', 'danger');
        }
    }
    
    // Kiểm tra prediction có hợp lệ với mode hiện tại không
    isValidPrediction(prediction) {
        if (prediction === '--' || prediction.includes('Lỗi')) {
            return false;
        }
        
        // Nếu prediction là 1 ký tự
        if (prediction.length === 1) {
            const char = prediction.toUpperCase();
            
            switch(this.currentMode) {
                case 'letters':
                    return /[A-Z]/.test(char);
                case 'numbers':
                    return /[0-9]/.test(char);
                case 'all':
                default:
                    return /[A-Z0-9]/.test(char);
            }
        }
        
        return true;
    }
    
    async sendToServer(imageData) {
        try {
            console.log(`📤 Sending request with mode: ${this.currentMode}`);
            const response = await fetch('/api/recognize/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: `image=${encodeURIComponent(imageData)}&mode=${this.currentMode}`
            });
            
            if (!response.ok) {
                console.error(`❌ HTTP error! status: ${response.status}`);
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('📥 Received response:', data);
            return data;
            
        } catch (error) {
            console.error('❌ Lỗi kết nối server:', error);
            return { success: false, prediction: 'Lỗi kết nối server', confidence: 0 };
        }
    }
    
    updateResult(prediction, confidence, handDetected = true) {
        if (!this.result || !this.confidence || !this.confidenceBar) return;
        
        // Show hand detection status
        if (!handDetected) {
            this.result.textContent = '👋';
            this.confidence.textContent = 'Không phát hiện bàn tay';
            this.confidenceBar.style.width = '0%';
            this.confidenceBar.style.background = '#6c757d';
            this.result.className = 'display-1 fw-bold text-muted mb-3';
            return;
        }
        
        this.result.textContent = prediction;
        this.confidence.textContent = `Độ tin cậy: ${confidence.toFixed(1)}%`;
        this.confidenceBar.style.width = `${confidence}%`;
        
        // Đổi màu thanh confidence
        if (confidence >= 80) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #0d6efd, #0dcaf0)'; // Xanh lam
            this.result.className = 'display-1 fw-bold text-primary mb-3 pulse-animation';
        } else if (confidence >= 60) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #198754, #20c997)'; // Xanh lá
            this.result.className = 'display-1 fw-bold text-success mb-3 pulse-animation';
        } else if (confidence > 0) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #dc3545, #e83e8c)'; // Đỏ
            this.result.className = 'display-1 fw-bold text-danger mb-3';
        } else {
            this.confidenceBar.style.background = '#e9ecef'; // Xám
            this.result.className = 'display-1 fw-bold text-secondary mb-3';
        }
        
        this.result.classList.add('pulse-animation');
        setTimeout(() => {
            if (this.result) this.result.classList.remove('pulse-animation');
        }, 500);
        
        if (prediction !== '--') {
            this.updateStatus(`Đã nhận diện: ${prediction}`);
        }
    }
    
    updateCaptureResult(prediction, confidence) {
        if (!this.captureResult || !this.captureConfidence || !this.captureConfidenceBar) return;
        
        this.captureResult.textContent = prediction;
        this.captureConfidence.textContent = `Độ tin cậy: ${confidence.toFixed(1)}%`;
        this.captureConfidenceBar.style.width = `${confidence}%`;
        
        // Đổi màu thanh confidence cho ảnh chụp
        if (confidence >= 80) {
            this.captureConfidenceBar.style.background = 'linear-gradient(90deg, #0dcaf0, #0d6efd)'; // Xanh lam đảo ngược
            this.captureResult.className = 'display-1 fw-bold text-info mb-3 pulse-animation';
        } else if (confidence >= 60) {
            this.captureConfidenceBar.style.background = 'linear-gradient(90deg, #20c997, #198754)'; // Xanh lá đảo ngược
            this.captureResult.className = 'display-1 fw-bold text-success mb-3 pulse-animation';
        } else if (confidence > 0) {
            this.captureConfidenceBar.style.background = 'linear-gradient(90deg, #e83e8c, #dc3545)'; // Đỏ đảo ngược
            this.captureResult.className = 'display-1 fw-bold text-danger mb-3';
        } else {
            this.captureConfidenceBar.style.background = '#e9ecef'; // Xám
            this.captureResult.className = 'display-1 fw-bold text-secondary mb-3';
        }
        
        this.captureResult.classList.add('pulse-animation');
        setTimeout(() => {
            if (this.captureResult) this.captureResult.classList.remove('pulse-animation');
        }, 500);
    }
    
    updateCaptureStatus(message, type) {
        if (this.captureStatus) {
            this.captureStatus.innerHTML = `<i class="fas fa-image me-2"></i>${message}`;
            this.captureStatus.className = `fs-5 text-${type}`;
        }
    }
    
    speakText() {
        const text = this.result ? this.result.textContent : '';
        this.speakTextContent(text);
    }
    
    speakCaptureText(text) {
        this.speakTextContent(text);
    }
    
    speakTextContent(text) {
        if (text && text !== '--' && text !== 'Lỗi nhận diện' && text !== 'Lỗi kết nối') {
            this.speechSynth.cancel();
            
            const utterance = new SpeechSynthesisUtterance(text);
            
            utterance.rate = 0.8;
            utterance.pitch = 1;
            utterance.volume = 1;
            
            const vietnameseVoice = this.voices.find(voice => 
                voice.lang.includes('vi') || voice.lang.includes('VN')
            );
            
            if (vietnameseVoice) {
                utterance.voice = vietnameseVoice;
                utterance.lang = 'vi-VN';
            } else {
                utterance.lang = 'en-US';
            }
            
            utterance.onstart = () => {
                if (this.speakBtn) {
                    this.speakBtn.innerHTML = '<i class="fas fa-volume-up me-2"></i>ĐANG ĐỌC...';
                    this.speakBtn.disabled = true;
                }
            };
            
            utterance.onend = () => {
                if (this.speakBtn) {
                    this.speakBtn.innerHTML = '<i class="fas fa-play-circle me-2"></i>ĐỌC KẾT QUẢ';
                    this.speakBtn.disabled = false;
                }
            };
            
            this.speechSynth.speak(utterance);
            console.log(`🔊 Phát âm: ${text}`);
        }
    }
    
    toggleAutoSpeech() {
        this.autoSpeech = !this.autoSpeech;
        
        if (this.autoSpeech) {
            this.speechStatus.textContent = 'BẬT';
            this.speechStatus.className = 'badge bg-success ms-2';
            this.toggleSpeechBtn.classList.remove('btn-outline-info');
            this.toggleSpeechBtn.classList.add('btn-info');
            this.updateStatus('Tự động phát âm đã BẬT');
        } else {
            this.speechStatus.textContent = 'TẮT';
            this.speechStatus.className = 'badge bg-secondary ms-2';
            this.toggleSpeechBtn.classList.remove('btn-info');
            this.toggleSpeechBtn.classList.add('btn-outline-info');
            this.updateStatus('Tự động phát âm đã TẮT');
        }
    }
    
    testSpeech() {
        const testText = "Xin chào! Hệ thống nhận diện ASL đã sẵn sàng";
        const utterance = new SpeechSynthesisUtterance(testText);
        
        utterance.rate = 0.8;
        utterance.volume = 1;
        
        this.speechSynth.speak(utterance);
        this.updateStatus('Đang kiểm tra âm thanh...');
        
        utterance.onend = () => {
            this.updateStatus('Kiểm tra âm thanh hoàn tất');
        };
    }
    
    updateStatus(message) {
        if (this.status) {
            this.status.innerHTML = `<i class="fas fa-circle text-primary me-2"></i>${message}`;
        }
    }
    
    getCSRFToken() {
        const name = 'csrftoken';
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    
    stopCamera() {
        console.log('🛑 Stopping camera and recognition...');
        
        if (this.recognitionInterval) {
            clearInterval(this.recognitionInterval);
            this.recognitionInterval = null;
        }
        
        this.speechSynth.cancel();
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
            this.stream = null;
        }
        
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        if (this.captureBtn) this.captureBtn.disabled = true;
        this.speakBtn.disabled = true;
        this.isRunning = false;
        
        // Reset real-time results
        if (this.result) {
            this.result.textContent = '--';
            this.confidence.textContent = 'Độ tin cậy: 0%';
            this.confidenceBar.style.width = '0%';
            this.confidenceBar.style.background = '#e9ecef';
            this.result.className = 'display-1 fw-bold text-primary mb-3';
        }
        
        this.updateStatus('Đã dừng nhận diện');
        
        // Reset capture section
        this.retakeImage();
    }
}

document.addEventListener('DOMContentLoaded', function() {
    console.log('📄 Page loaded, initializing ASL Real Recognizer with Capture Feature...');
    
    // Kiểm tra các element cần thiết
    const requiredElements = [
        'video', 'canvas', 'capture-canvas', 'start-btn', 'stop-btn',
        'prediction-result', 'confidence', 'confidence-bar', 'status'
    ];
    
    let allElementsFound = true;
    requiredElements.forEach(elementId => {
        const element = document.getElementById(elementId);
        if (!element) {
            console.error(`❌ Missing element: ${elementId}`);
            allElementsFound = false;
        }
    });
    
    if (!allElementsFound) {
        console.error('❌ Some required elements are missing. Please check the HTML.');
        return;
    }
    
    try {
        const recognizer = new ASLRealRecognizer();
        console.log('✅ ASL Real Recognizer initialized successfully');
    } catch (error) {
        console.error('❌ Error initializing ASL Real Recognizer:', error);
    }
});