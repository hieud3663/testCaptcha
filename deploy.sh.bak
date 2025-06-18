#!/bin/bash

# Cấu hình thông tin dự án
PROJECT_ID="captcha-185046619914"  # Thay thế bằng GCP Project ID của bạn
SERVICE_NAME="captcha-solver-api"
REGION="asia-southeast1"  # Có thể thay đổi region phù hợp
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Triển khai Captcha Solver API lên Google Cloud Run"
echo "📋 Thông tin:"
echo "  - Project ID: $PROJECT_ID"
echo "  - Service Name: $SERVICE_NAME"
echo "  - Region: $REGION"
echo "  - Image: $IMAGE_NAME"
echo

# 1. Kiểm tra gcloud CLI
echo "1️⃣ Kiểm tra gcloud CLI..."
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI chưa được cài đặt!"
    echo "Hướng dẫn cài đặt: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# 2. Kiểm tra Docker
echo "2️⃣ Kiểm tra Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker chưa được cài đặt!"
    echo "Hướng dẫn cài đặt: https://docs.docker.com/get-docker/"
    exit 1
fi

# 3. Xác thực với Google Cloud
echo "3️⃣ Xác thực với Google Cloud..."
gcloud auth login
gcloud config set project $PROJECT_ID

# 4. Enable APIs cần thiết
echo "4️⃣ Enable Google Cloud APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com

# 5. Configure Docker để sử dụng gcloud
echo "5️⃣ Cấu hình Docker..."
gcloud auth configure-docker

# 6. Build Docker image
echo "6️⃣ Build Docker image..."
docker build -t $IMAGE_NAME .

# 7. Push image lên Google Container Registry
echo "7️⃣ Push image lên Google Container Registry..."
docker push $IMAGE_NAME

# 8. Deploy lên Cloud Run
echo "8️⃣ Deploy lên Google Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --concurrency 10 \
    --port 8080

# 9. Lấy URL của service
echo "9️⃣ Lấy thông tin service..."
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')

echo
echo "🎉 TRIỂN KHAI THÀNH CÔNG!"
echo "🌐 Service URL: $SERVICE_URL"
echo
echo "📖 Test API:"
echo "Health Check:"
echo "curl $SERVICE_URL/health"
echo
echo "Solve Captcha:"
echo "curl -X POST $SERVICE_URL/solve-captcha \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"image\": \"YOUR_BASE64_IMAGE\"}'"
echo
echo "🔧 Để cập nhật service, chạy lại script này"
echo "🗑️ Để xóa service: gcloud run services delete $SERVICE_NAME --region $REGION"
