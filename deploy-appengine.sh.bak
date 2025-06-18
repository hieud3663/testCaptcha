#!/bin/bash

# Cấu hình thông tin dự án
PROJECT_ID="captcha-185046619914"  # Thay thế bằng GCP Project ID của bạn
SERVICE_NAME="captcha-solver-api"

echo "🚀 Triển khai Captcha Solver API lên Google App Engine"
echo "📋 Project ID: $PROJECT_ID"
echo

# 1. Kiểm tra gcloud CLI
echo "1️⃣ Kiểm tra gcloud CLI..."
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI chưa được cài đặt!"
    echo "Hướng dẫn cài đặt: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# 2. Xác thực với Google Cloud
echo "2️⃣ Xác thực với Google Cloud..."
gcloud auth login
gcloud config set project $PROJECT_ID

# 3. Enable APIs cần thiết
echo "3️⃣ Enable Google Cloud APIs..."
gcloud services enable \
    appengine.googleapis.com \
    cloudbuild.googleapis.com

# 4. Tạo App Engine application (nếu chưa có)
echo "4️⃣ Tạo App Engine application..."
gcloud app create --region=asia-southeast1 || echo "App Engine đã tồn tại"

# 5. Deploy lên App Engine
echo "5️⃣ Deploy lên Google App Engine..."
gcloud app deploy app.yaml --quiet

# 6. Lấy URL của app
echo "6️⃣ Lấy thông tin app..."
APP_URL=$(gcloud app browse --no-launch-browser)

echo
echo "🎉 TRIỂN KHAI THÀNH CÔNG!"
echo "🌐 App URL: https://$PROJECT_ID.appspot.com"
echo
echo "📖 Test API:"
echo "Health Check:"
echo "curl https://$PROJECT_ID.appspot.com/health"
echo
echo "Solve Captcha:"
echo "curl -X POST https://$PROJECT_ID.appspot.com/solve-captcha \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"image\": \"YOUR_BASE64_IMAGE\"}'"
echo
echo "🔧 Để cập nhật app, chạy lại script này"
echo "🗑️ Để xóa app: gcloud app versions delete [VERSION]"
