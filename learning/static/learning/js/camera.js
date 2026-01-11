// learning/static/learning/js/camera.js
class ASLRecognizer {
    constructor() {
        this.video = document.getElementById('video');
        this.startBtn = document.getElementById('start-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.result = document.getElementById('prediction-result');
        this.confidence = document.getElementById('confidence');
        this.confidenceBar = document.getElementById('confidence-bar');
        this.stream = null;
        this.isRunning = false;
        this.recognizer = null;
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
    }
    
    async startCamera() {
        try {
            console.log('🚀 Starting camera and ASL recognition...');
            
            // Khởi động camera
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: 640, 
                    height: 480,
                    facingMode: 'user'
                } 
            });
            
            this.video.srcObject = this.stream;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.isRunning = true;
            
            // Khởi tạo nhận diện ASL
            await this.initializeASLRecognition();
            
        } catch (error) {
            console.error('❌ Lỗi khi truy cập camera:', error);
            alert('Không thể truy cập camera. Vui lòng kiểm tra quyền truy cập và thử lại.');
        }
    }
    
    async initializeASLRecognition() {
        try {
            // Giả lập kết nối đến model AI (sẽ thay bằng AI thật sau)
            console.log('🔮 Initializing ASL recognition model...');
            
            // Bắt đầu nhận diện
            this.startRecognition();
            
        } catch (error) {
            console.error('❌ Lỗi khi khởi tạo nhận diện ASL:', error);
            this.startSimulation(); // Fallback về mô phỏng nếu lỗi
        }
    }
    
    startRecognition() {
        console.log('🎯 Starting real-time ASL recognition...');
        
        // TẠM THỜI: Sử dụng mô phỏng nhận diện
        // SAU NÀY: Sẽ tích hợp model AI thật từ realtime_recognition_v1.py
        this.startSimulation();
    }
    
    startSimulation() {
        // Mô phỏng nhận diện ASL - SAU SẼ THAY BẰNG AI THẬT
        const aslAlphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
        const aslWords = ['hello', 'thank you', 'please', 'sorry', 'help', 'love', 'family', 'friend'];
        const allSigns = [...aslAlphabet.split(''), ...aslWords];
        
        console.log('🔍 ASL recognition simulation started...');
        
        const recognize = () => {
            if (this.isRunning) {
                // Giả lập nhận diện ngẫu nhiên
                const randomSign = allSigns[Math.floor(Math.random() * allSigns.length)];
                const randomConfidence = (Math.random() * 40 + 60).toFixed(1); // 60-100%
                
                // Hiển thị kết quả
                this.updateResult(randomSign, randomConfidence);
                
                // Tiếp tục nhận diện
                setTimeout(recognize, 1500);
            }
        };
        
        recognize();
    }
    
    updateResult(prediction, confidence) {
        // Cập nhật kết quả nhận diện
        this.result.textContent = prediction;
        this.confidence.textContent = `Độ tin cậy: ${confidence}%`;
        this.confidenceBar.style.width = `${confidence}%`;
        
        // Đổi màu thanh confidence
        if (confidence >= 80) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #28a745, #20c997)';
        } else if (confidence >= 60) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #ffc107, #fd7e14)';
        } else {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #dc3545, #e83e8c)';
        }
        
        console.log(`✅ Nhận diện: ${prediction} (${confidence}%)`);
    }
    
    stopCamera() {
        console.log('🛑 Stopping camera and recognition...');
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
        }
        
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.isRunning = false;
        
        // Reset hiển thị
        this.result.textContent = '--';
        this.confidence.textContent = 'Độ tin cậy: 0%';
        this.confidenceBar.style.width = '0%';
    }
}

// Khởi tạo khi trang được load
document.addEventListener('DOMContentLoaded', function() {
    console.log('📄 Page loaded, initializing ASL Recognizer...');
    new ASLRecognizer();
});

