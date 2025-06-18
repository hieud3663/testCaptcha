#!/bin/bash

echo "🚀 HƯỚNG DẪN TRIỂN KHAI CAPTCHA SOLVER API LÊN GOOGLE CLOUD"
echo "============================================================="
echo

# Kiểm tra môi trường
echo "📋 BƯỚC 1: KIỂM TRA MÔI TRƯỜNG"
echo "------------------------------"

# Check gcloud
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud SDK chưa được cài đặt!"
    echo
    echo "Cài đặt Google Cloud SDK:"
    echo "curl https://sdk.cloud.google.com | bash"
    echo "exec -l \$SHELL"
    echo "gcloud init"
    echo
    exit 1
else
    echo "✅ Google Cloud SDK: $(gcloud --version | head -n1)"
fi

# Check docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker chưa được cài đặt!"
    echo
    echo "Cài đặt Docker:"
    echo "curl -fsSL https://get.docker.com -o get-docker.sh"
    echo "sh get-docker.sh"
    echo
    exit 1
else
    echo "✅ Docker: $(docker --version)"
fi

echo

# Hướng dẫn cấu hình
echo "📋 BƯỚC 2: CẤU HÌNH GOOGLE CLOUD PROJECT"
echo "----------------------------------------"
echo
echo "1. Tạo hoặc chọn Google Cloud Project:"
echo "   - Truy cập: https://console.cloud.google.com"
echo "   - Tạo project mới hoặc chọn project có sẵn"
echo "   - Ghi nhớ PROJECT_ID"
echo
echo "2. Enable Billing cho project:"
echo "   - Project phải có billing account được enable"
echo "   - Cloud Run có free tier 2 triệu requests/tháng"
echo
echo "3. Đăng nhập và cấu hình gcloud:"
echo "   gcloud auth login"
echo "   gcloud config set project YOUR_PROJECT_ID"
echo
echo "4. Enable APIs cần thiết:"
echo "   gcloud services enable cloudbuild.googleapis.com"
echo "   gcloud services enable run.googleapis.com"
echo "   gcloud services enable containerregistry.googleapis.com"
echo

# Hướng dẫn deploy
echo "📋 BƯỚC 3: TRIỂN KHAI"
echo "--------------------"
echo
echo "PHƯƠNG PHÁP 1: Google Cloud Run (Khuyến nghị)"
echo "============================================="
echo
echo "1. Chỉnh sửa PROJECT_ID trong deploy.sh:"
echo "   nano deploy.sh"
echo "   # Thay 'your-project-id' bằng PROJECT_ID thực tế"
echo
echo "2. Chạy deployment:"
echo "   chmod +x deploy.sh"
echo "   ./deploy.sh"
echo
echo "3. Sau khi deploy thành công, bạn sẽ nhận được URL:"
echo "   https://captcha-solver-api-xxxxx-as.a.run.app"
echo
echo
echo "PHƯƠNG PHÁP 2: Google App Engine"
echo "================================"
echo
echo "1. Chỉnh sửa PROJECT_ID trong deploy-appengine.sh:"
echo "   nano deploy-appengine.sh"
echo
echo "2. Deploy:"
echo "   chmod +x deploy-appengine.sh"
echo "   ./deploy-appengine.sh"
echo
echo "3. URL sẽ là:"
echo "   https://YOUR_PROJECT_ID.appspot.com"
echo

# Test local
echo "📋 BƯỚC 4: TEST TRƯỚC KHI DEPLOY"
echo "--------------------------------"
echo
echo "1. Test Docker locally:"
echo "   chmod +x test-docker.sh"
echo "   ./test-docker.sh"
echo
echo "2. Test API endpoints:"
echo "   # Health check"
echo "   curl http://localhost:8080/health"
echo
echo "   # Solve captcha (cần file image.png)"
echo "   python client.py solve http://localhost:8080 image.png"
echo

# Sau deploy
echo "📋 BƯỚC 5: SAU KHI DEPLOY"
echo "------------------------"
echo
echo "1. Test API trên cloud:"
echo "   curl https://your-service-url/health"
echo
echo "2. Sử dụng Python client:"
echo "   from client import CaptchaSolverClient"
echo "   client = CaptchaSolverClient('https://your-service-url')"
echo "   result = client.solve_captcha(image_path='captcha.png')"
echo
echo "3. Monitor service:"
echo "   - Cloud Console > Cloud Run/App Engine"
echo "   - Xem logs: gcloud logs read --service=captcha-solver-api"
echo

# Troubleshooting
echo "📋 TROUBLESHOOTING"
echo "------------------"
echo
echo "❌ Build thất bại:"
echo "   - Kiểm tra Dockerfile syntax"
echo "   - Đảm bảo requirements.txt đúng"
echo "   - Kiểm tra network connection"
echo
echo "❌ Deploy thất bại:"
echo "   - Kiểm tra PROJECT_ID đúng"
echo "   - Đảm bảo billing enabled"
echo "   - Kiểm tra APIs enabled"
echo "   - Xem logs: gcloud logs read"
echo
echo "❌ API không response:"
echo "   - Kiểm tra PORT environment variable"
echo "   - Test health endpoint trước"
echo "   - Xem logs container"
echo
echo "❌ Memory/Timeout issues:"
echo "   - Tăng memory: --memory 4Gi"
echo "   - Tăng timeout: --timeout 600"
echo "   - Tăng CPU: --cpu 2"
echo

# Chi phí
echo "📋 CHI PHÍ ƯỚC TÍNH"
echo "-------------------"
echo
echo "Google Cloud Run:"
echo "- Free tier: 2 triệu requests/tháng"
echo "- Sau đó: ~\$0.40/triệu requests"
echo "- Memory: ~\$0.0000025/GB-second"
echo "- CPU: ~\$0.00001/vCPU-second"
echo
echo "Ví dụ: 100k requests/tháng với 2GB RAM:"
echo "- Requests: Free (< 2M)"
echo "- Memory: ~\$3-5/tháng"
echo "- Tổng: ~\$3-5/tháng"
echo

echo "🎉 CHÚC BẠN DEPLOY THÀNH CÔNG!"
echo "=============================="
echo
echo "📞 Hỗ trợ:"
echo "- Google Cloud Documentation: https://cloud.google.com/run/docs"
echo "- Pricing Calculator: https://cloud.google.com/products/calculator"
echo "- Community Support: https://stackoverflow.com/questions/tagged/google-cloud-run"
