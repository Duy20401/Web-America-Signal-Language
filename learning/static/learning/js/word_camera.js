// learning/static/learning/js/word_camera.js
class ASLWordRecognizer {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.startBtn = document.getElementById('start-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.resetBtn = document.getElementById('reset-btn');
        this.speakBtn = document.getElementById('speak-btn');
        this.toggleSpeechBtn = document.getElementById('toggle-speech-btn');
        this.testSpeechBtn = document.getElementById('test-speech-btn');
        this.result = document.getElementById('prediction-result');
        this.confidence = document.getElementById('confidence');
        this.confidenceBar = document.getElementById('confidence-bar');
        this.status = document.getElementById('status');
        this.speechStatus = document.getElementById('speech-status');
        
        // Progress elements
        this.bufferStatus = document.getElementById('buffer-status');
        this.progressText = document.getElementById('progress-text');
        this.progressPercent = document.getElementById('progress-percent');
        this.collectionInfo = document.getElementById('collection-info');
        this.recognitionState = document.getElementById('recognition-state');
        this.processingTime = document.getElementById('processing-time');
        
        this.stream = null;
        this.isRunning = false;
        this.autoSpeech = false;
        this.lastPrediction = '';
        this.recognitionInterval = null;
        this.lastProcessTime = 0;
        
        this.speechSynth = window.speechSynthesis;
        this.voices = [];
        
        // Thiết lập kích thước canvas
        this.canvas.width = 640;
        this.canvas.height = 480;
        
        this.initializeEventListeners();
        this.loadVoices();
    }
    
    initializeEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        this.resetBtn.addEventListener('click', () => this.resetRecognition());
        this.speakBtn.addEventListener('click', () => this.speakText());
        this.toggleSpeechBtn.addEventListener('click', () => this.toggleAutoSpeech());
        this.testSpeechBtn.addEventListener('click', () => this.testSpeech());
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
            console.log('🚀 Starting word recognition camera...');
            this.updateStatus('Đang khởi động camera...');
            this.updateRecognitionState('Khởi động');
            
            if (!this.video) {
                console.error('❌ Video element not found');
                this.updateStatus('Lỗi: Không tìm thấy video element');
                return;
            }
            
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user',
                    frameRate: { ideal: 30 }
                }
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            
            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    console.log('✅ Camera metadata loaded');
                    resolve();
                };
                
                this.video.onloadeddata = () => {
                    console.log('✅ Camera data loaded');
                    resolve();
                };
                
                setTimeout(resolve, 1000);
            });
            
            await this.video.play();
            
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.resetBtn.disabled = false;
            this.speakBtn.disabled = false;
            this.isRunning = true;
            
            console.log('✅ Camera started successfully');
            this.updateStatus('Camera đã sẵn sàng. Đang nhận diện từ vựng...');
            this.updateRecognitionState('Thu thập dữ liệu');
            this.startRecognition();
            
        } catch (error) {
            console.error('❌ Lỗi khi truy cập camera:', error);
            this.updateStatus('Lỗi: Không thể truy cập camera');
            this.updateRecognitionState('Lỗi camera');
            
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
        console.log('🎯 Starting word recognition...');
        this.updateStatus('AI đang nhận diện từ vựng...');
        this.updateRecognitionState('Thu thập dữ liệu');
        
        const checkVideoReady = () => {
            if (this.video.readyState >= this.video.HAVE_ENOUGH_DATA) {
                console.log('✅ Video ready for recognition');
                this.recognitionInterval = setInterval(() => {
                    if (this.isRunning) {
                        this.captureAndRecognize();
                    }
                }, 800); // 1.25 FPS để đủ thời gian xử lý
            } else {
                console.log('⏳ Waiting for video to be ready...');
                setTimeout(checkVideoReady, 100);
            }
        };
        
        checkVideoReady();
    }
    
    async captureAndRecognize() {
        try {
            if (this.video.readyState < this.video.HAVE_ENOUGH_DATA) {
                return;
            }
            
            const startTime = performance.now();
            
            this.ctx.save();
            this.ctx.scale(-1, 1);
            this.ctx.drawImage(this.video, -this.canvas.width, 0, this.canvas.width, this.canvas.height);
            this.ctx.restore();
            
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
            const response = await this.sendToServer(imageData);
            
            const processTime = performance.now() - startTime;
            this.lastProcessTime = processTime;
            this.updateProcessingTime(processTime);
            
            if (response.success) {
                this.updateResult(
                    response.prediction, 
                    response.confidence, 
                    response.buffer_status
                );
                
                if (this.autoSpeech && response.confidence > 50 && 
                    response.prediction !== this.lastPrediction &&
                    response.prediction !== '--' &&
                    !response.prediction.includes('Lỗi') &&
                    !response.prediction.includes('thu thập') &&
                    !response.prediction.includes('đang thu thập')) {
                    this.speakText();
                }
                
                this.lastPrediction = response.prediction;
            } else {
                this.updateResult(response.prediction || 'Lỗi nhận diện', 0);
            }
            
        } catch (error) {
            console.error('❌ Lỗi nhận diện từ:', error);
            this.updateResult('Lỗi kết nối', 0);
            this.updateRecognitionState('Lỗi kết nối');
        }
    }
    
    async sendToServer(imageData) {
        try {
            const response = await fetch('/api/recognize/words/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: `image=${encodeURIComponent(imageData)}`
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
            
        } catch (error) {
            console.error('❌ Lỗi kết nối server:', error);
            return { success: false, prediction: 'Lỗi kết nối server', confidence: 0 };
        }
    }
    
    async resetRecognition() {
        console.log('🔄 Resetting recognition...');
        this.updateStatus('Đang reset hệ thống nhận diện...');
        this.updateRecognitionState('Resetting');
        
        try {
            const response = await fetch('/api/recognize/words/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: 'reset=true'
            });
            
            if (response.ok) {
                this.updateResult('--', 0);
                this.updateProgress(0, 15);
                this.updateStatus('Đã reset. Bắt đầu cử chỉ mới...');
                this.updateRecognitionState('Sẵn sàng');
                console.log('✅ Reset successful');
            }
        } catch (error) {
            console.error('❌ Reset error:', error);
            this.updateStatus('Lỗi reset hệ thống');
        }
    }
    
    updateResult(prediction, confidence, bufferInfo = null) {
        this.result.textContent = prediction;
        this.confidence.textContent = `Độ tin cậy: ${confidence.toFixed(1)}%`;
        this.confidenceBar.style.width = `${confidence}%`;
        
        // Cập nhật màu sắc confidence bar
        if (confidence >= 80) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #28a745, #20c997)';
            this.result.className = 'display-1 fw-bold text-success mb-3 pulse-animation';
            this.updateRecognitionState('Nhận diện thành công');
        } else if (confidence >= 60) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #ffc107, #fd7e14)';
            this.result.className = 'display-1 fw-bold text-warning mb-3 pulse-animation';
            this.updateRecognitionState('Nhận diện khá');
        } else if (confidence > 0) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #dc3545, #e83e8c)';
            this.result.className = 'display-1 fw-bold text-danger mb-3';
            this.updateRecognitionState('Độ tin cậy thấp');
        } else {
            this.confidenceBar.style.background = '#e9ecef';
            this.result.className = 'display-1 fw-bold text-secondary mb-3';
        }
        
        // Hiển thị trạng thái buffer nếu có
        if (bufferInfo) {
            this.updateProgress(bufferInfo.current_size, bufferInfo.required_size);
            
            if (bufferInfo.current_size < bufferInfo.required_size) {
                this.updateRecognitionState('Thu thập dữ liệu');
                this.collectionInfo.textContent = `Đang thu thập dữ liệu cử chỉ... (${bufferInfo.current_size}/${bufferInfo.required_size})`;
            } else {
                this.updateRecognitionState('Đang phân tích');
                this.collectionInfo.textContent = 'Đủ dữ liệu. Đang phân tích cử chỉ...';
            }
        }
        
        this.result.classList.add('pulse-animation');
        setTimeout(() => {
            this.result.classList.remove('pulse-animation');
        }, 500);
        
        if (prediction !== '--' && !prediction.includes('Lỗi')) {
            this.updateStatus(`Đã nhận diện: ${prediction}`);
        }
    }
    
    updateProgress(current, total) {
        const progressPercent = Math.min(100, (current / total) * 100);
        
        this.bufferStatus.style.width = `${progressPercent}%`;
        this.progressText.textContent = `${current}/${total} frame`;
        this.progressPercent.textContent = `${Math.round(progressPercent)}%`;
        
        if (progressPercent < 100) {
            this.bufferStatus.className = 'progress-bar progress-bar-striped progress-bar-animated bg-warning';
        } else {
            this.bufferStatus.className = 'progress-bar progress-bar-striped progress-bar-animated bg-success';
        }
    }
    
    updateRecognitionState(state) {
        if (this.recognitionState) {
            this.recognitionState.textContent = state;
            
            // Thêm màu sắc cho trạng thái
            const stateElement = this.recognitionState;
            stateElement.className = '';
            
            if (state.includes('thành công')) {
                stateElement.classList.add('text-success', 'fw-bold');
            } else if (state.includes('Lỗi')) {
                stateElement.classList.add('text-danger', 'fw-bold');
            } else if (state.includes('thu thập')) {
                stateElement.classList.add('text-warning', 'fw-bold');
            } else {
                stateElement.classList.add('text-info', 'fw-bold');
            }
        }
    }
    
    updateProcessingTime(time) {
        if (this.processingTime) {
            this.processingTime.textContent = `${time.toFixed(1)}ms`;
        }
    }
    
    speakText() {
        const text = this.result.textContent;
        
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
                this.speakBtn.innerHTML = '<i class="fas fa-volume-up me-2"></i>ĐANG ĐỌC...';
                this.speakBtn.disabled = true;
            };
            
            utterance.onend = () => {
                this.speakBtn.innerHTML = '<i class="fas fa-play-circle me-2"></i>ĐỌC KẾT QUẢ';
                this.speakBtn.disabled = false;
            };
            
            utterance.onerror = (event) => {
                console.error('❌ Speech synthesis error:', event.error);
                this.speakBtn.innerHTML = '<i class="fas fa-play-circle me-2"></i>ĐỌC KẾT QUẢ';
                this.speakBtn.disabled = false;
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
        const testText = "Xin chào! Hệ thống nhận diện từ vựng ASL đã sẵn sàng";
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
        console.log('🛑 Stopping word recognition...');
        
        if (this.recognitionInterval) {
            clearInterval(this.recognitionInterval);
            this.recognitionInterval = null;
        }
        
        this.speechSynth.cancel();
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => {
                track.stop();
            });
            this.video.srcObject = null;
            this.stream = null;
        }
        
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.resetBtn.disabled = true;
        this.speakBtn.disabled = true;
        this.isRunning = false;
        
        this.result.textContent = '--';
        this.confidence.textContent = 'Độ tin cậy: 0%';
        this.confidenceBar.style.width = '0%';
        this.confidenceBar.style.background = '#e9ecef';
        this.result.className = 'display-1 fw-bold text-primary mb-3';
        this.updateProgress(0, 15);
        this.updateStatus('Đã dừng nhận diện');
        this.updateRecognitionState('Đã dừng');
        this.updateProcessingTime(0);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    console.log('📄 Word recognition page loaded - COMPLETE VERSION');
    new ASLWordRecognizer();
});