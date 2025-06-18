# 🚀 Hướng Dẫn Triển Khai Google Cloud Từ Đầu

## 📋 Tổng Quan

Hướng dẫn này sẽ giúp bạn triển khai Captcha Solver API lên Google Cloud từ con số 0, ngay cả khi bạn chưa từng sử dụng Google Cloud trước đây.

**Thời gian ước tính:** 30-45 phút  
**Chi phí:** Miễn phí với free tier (2 triệu requests/tháng)

## 🎯 Yêu Cầu

- ✅ Tài khoản Google (Gmail)
- ✅ Thẻ tín dụng/ghi nợ (để verify, không bị charge)
- ✅ Máy tính có internet
- ✅ Terminal/Command line access

## 📋 Bước 1: Chuẩn Bị Google Cloud Account

### 1.1 Đăng ký Google Cloud Platform

1. **Truy cập:** https://cloud.google.com/
2. **Click "Get started for free"**
3. **Đăng nhập** bằng tài khoản Google
4. **Nhập thông tin:**
   - Quốc gia/vùng: Vietnam
   - Loại tài khoản: Individual
   - Thông tin thanh toán (cần thiết để verify)
5. **Nhận $300 credit miễn phí** (300 USD cho tài khoản mới)

### 1.2 Tạo Project Đầu Tiên

1. **Truy cập:** https://console.cloud.google.com/
2. **Click dropdown** ở góc trên bên trái
3. **Click "New Project"**
4. **Nhập thông tin:**
   - Project name: `Captcha Solver API`
   - Project ID: `captcha-solver-[số-random]` (phải unique)
5. **Click "Create"**

## 📋 Bước 2: Cài Đặt Google Cloud SDK

### 2.1 Tự Động (Khuyến nghị)

Chạy script tự động:

```bash
cd /workspaces/bot3663/testCaptcha
./full-setup.sh
```

Script này sẽ:
- ✅ Cài đặt Google Cloud SDK
- ✅ Đăng nhập và cấu hình
- ✅ Enable APIs cần thiết
- ✅ Deploy API automatically

### 2.2 Thủ Công (Nếu cần)

#### Linux/macOS:
```bash
# Download và cài đặt
curl https://sdk.cloud.google.com | bash

# Restart shell
exec -l $SHELL

# Initialize
gcloud init
```

#### Windows:
1. Download từ: https://cloud.google.com/sdk/docs/install-sdk
2. Chạy installer
3. Mở Command Prompt
4. Chạy: `gcloud init`

## 📋 Bước 3: Cấu Hình Project

### 3.1 Đăng Nhập

```bash
# Đăng nhập (browser sẽ mở)
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### 3.2 Enable Billing

⚠️ **Quan trọng:** Cần enable billing để sử dụng Cloud Run

1. **Truy cập:** https://console.cloud.google.com/billing
2. **Link billing account** với project
3. **Hoặc tạo billing account mới**

**Đừng lo về chi phí:** Free tier có 2 triệu requests/tháng miễn phí!

### 3.3 Enable APIs

```bash
# Enable các APIs cần thiết
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## 📋 Bước 4: Deploy API

### 4.1 Cấu Hình Project ID

Chỉnh sửa file `deploy.sh`:

```bash
nano deploy.sh
```

Thay đổi dòng:
```bash
PROJECT_ID="your-project-id"  # Thay bằng PROJECT_ID thực tế
```

### 4.2 Deploy

```bash
# Chạy deployment
chmod +x deploy.sh
./deploy.sh
```

Quá trình deploy sẽ:
1. **Build Docker image** (~5 phút)
2. **Push lên Container Registry** (~2 phút)
3. **Deploy lên Cloud Run** (~3 phút)

### 4.3 Lấy URL

Sau khi deploy thành công, bạn sẽ nhận được URL như:
```
https://captcha-solver-api-xxxxx-as.a.run.app
```

## 📋 Bước 5: Test API

### 5.1 Health Check

```bash
curl https://your-service-url/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "service": "Captcha Puzzle Solver",
  "version": "1.0.0"
}
```

### 5.2 Test với Python Client

```bash
# Sử dụng client library
python client.py health https://your-service-url

# Test solve captcha
python client.py solve https://your-service-url image.png
```

### 5.3 Test với cURL

```bash
# Prepare base64 image
BASE64_IMAGE=$(base64 -w 0 image.png)

# Call API
curl -X POST https://your-service-url/solve-captcha \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$BASE64_IMAGE\"}"
```

## 🔧 Troubleshooting

### ❌ "gcloud: command not found"

**Solution:**
```bash
# Add to PATH
export PATH="$HOME/google-cloud-sdk/bin:$PATH"
echo 'export PATH="$HOME/google-cloud-sdk/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### ❌ "Billing account required"

**Solution:**
1. Truy cập: https://console.cloud.google.com/billing
2. Create hoặc link billing account
3. Verify bằng thẻ tín dụng

### ❌ "API not enabled"

**Solution:**
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
```

### ❌ "Permission denied"

**Solution:**
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login
```

### ❌ "Docker build failed"

**Solution:**
```bash
# Configure Docker
gcloud auth configure-docker

# Retry build
docker build -t test-image .
```

## 💰 Chi Phí

### Free Tier (Permanent)
- **Cloud Run:** 2 triệu requests/tháng
- **Cloud Build:** 120 phút build/ngày
- **Container Registry:** 0.5GB storage

### Paid Usage (sau free tier)
- **Requests:** ~$0.40/triệu requests
- **Memory:** ~$0.0000025/GB-second
- **CPU:** ~$0.00001/vCPU-second

### Ví Dụ Chi Phí Thực Tế
- **100K requests/tháng:** ~$3-5/tháng
- **1M requests/tháng:** ~$10-15/tháng
- **10M requests/tháng:** ~$50-80/tháng

## 📊 Monitoring & Management

### Cloud Console
- **URL:** https://console.cloud.google.com/
- **Cloud Run:** https://console.cloud.google.com/run
- **Logs:** https://console.cloud.google.com/logs

### Command Line
```bash
# View logs
gcloud logs read --service=captcha-solver-api

# List services
gcloud run services list

# Update service
gcloud run deploy captcha-solver-api --image=gcr.io/PROJECT_ID/captcha-solver-api

# Delete service
gcloud run services delete captcha-solver-api
```

## 🔒 Security Best Practices

### 1. API Authentication
```bash
# Disable public access
gcloud run services update captcha-solver-api --no-allow-unauthenticated

# Create service account
gcloud iam service-accounts create captcha-api-user
```

### 2. Environment Variables
```bash
# Set secrets
gcloud run services update captcha-solver-api \
  --set-env-vars="API_KEY=your-secret-key"
```

### 3. Rate Limiting
Implement trong code hoặc sử dụng Cloud Armor.

## 📱 Tích Hợp Vào Ứng Dụng

### Web Application
```javascript
// JavaScript example
async function solveCaptcha(imageFile) {
    const formData = new FormData();
    const reader = new FileReader();
    
    reader.onload = async (e) => {
        const base64 = e.target.result.split(',')[1];
        
        const response = await fetch('https://your-api-url/solve-captcha', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64 })
        });
        
        const result = await response.json();
        console.log('Coordinates:', result.coordinates);
    };
    
    reader.readAsDataURL(imageFile);
}
```

### Mobile App (React Native)
```javascript
import { launchImageLibrary } from 'react-native-image-picker';

const solveCaptcha = () => {
    launchImageLibrary({}, (response) => {
        if (response.assets) {
            const base64 = response.assets[0].base64;
            
            fetch('https://your-api-url/solve-captcha', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64 })
            })
            .then(res => res.json())
            .then(data => console.log(data.coordinates));
        }
    });
};
```

### Python Application
```python
from client import CaptchaSolverClient

client = CaptchaSolverClient("https://your-api-url")
coordinates = client.solve_captcha(image_path="captcha.png")

if coordinates:
    x, y = coordinates
    print(f"Captcha solved: ({x}, {y})")
```

## 🎉 Hoàn Thành!

Chúc mừng! Bạn đã successfully deploy Captcha Solver API lên Google Cloud. 

**Next Steps:**
1. ✅ Test API với traffic thực tế
2. ✅ Monitor usage và performance
3. ✅ Implement authentication nếu cần
4. ✅ Scale up theo nhu cầu

**Support:**
- 📧 Google Cloud Support
- 📚 Documentation: https://cloud.google.com/run/docs
- 💬 Stack Overflow: google-cloud-run tag
