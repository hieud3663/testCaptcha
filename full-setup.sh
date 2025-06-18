#!/bin/bash

echo "🚀 HƯỚNG DẪN TRIỂN KHAI GOOGLE CLOUD TỪ ĐẦU"
echo "================================================"
echo "📅 Ngày: $(date)"
echo

# Bước 1: Cài đặt Google Cloud SDK
echo "📋 BƯỚC 1: CÀI ĐẶT GOOGLE CLOUD SDK"
echo "====================================="
echo

if command -v gcloud &> /dev/null; then
    echo "✅ Google Cloud SDK đã được cài đặt"
    echo "   Version: $(gcloud --version | head -n1)"
else
    echo "❌ Google Cloud SDK chưa được cài đặt"
    echo
    echo "Đang cài đặt Google Cloud SDK..."
    
    # Cài đặt cho Linux/macOS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "🐧 Detected Linux - Installing..."
        
        # Download và cài đặt
        curl https://sdk.cloud.google.com > install.sh
        bash install.sh --disable-prompts
        
        # Add to PATH
        echo 'export PATH="$HOME/google-cloud-sdk/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🍎 Detected macOS - Installing..."
        
        # Sử dụng Homebrew nếu có
        if command -v brew &> /dev/null; then
            brew install --cask google-cloud-sdk
        else
            # Manual install
            curl https://sdk.cloud.google.com > install.sh
            bash install.sh --disable-prompts
        fi
    else
        echo "💻 For Windows, please download from:"
        echo "   https://cloud.google.com/sdk/docs/install-sdk"
        exit 1
    fi
    
    echo "✅ Google Cloud SDK cài đặt hoàn tất!"
fi

echo
echo "📋 BƯỚC 2: ĐĂNG NHẬP VÀ CẤU HÌNH"
echo "================================"
echo

# Đăng nhập
echo "🔑 Đăng nhập vào Google Cloud..."
echo "   Browser sẽ mở ra để bạn đăng nhập"
echo "   Chọn account Google có quyền truy cập Cloud"
echo

read -p "   Nhấn Enter để tiếp tục đăng nhập..."
gcloud auth login

if [ $? -eq 0 ]; then
    echo "✅ Đăng nhập thành công!"
else
    echo "❌ Đăng nhập thất bại! Vui lòng thử lại"
    exit 1
fi

echo
echo "📋 BƯỚC 3: TẠO HOẶC CHỌN PROJECT"
echo "================================"
echo

# List projects
echo "📋 Danh sách projects hiện có:"
gcloud projects list --format="table(projectId,name,projectNumber)"

echo
echo "Bạn có thể:"
echo "1. Sử dụng project có sẵn"
echo "2. Tạo project mới"
echo

read -p "Chọn (1/2): " choice

if [ "$choice" = "2" ]; then
    echo
    echo "🆕 Tạo project mới..."
    
    read -p "Nhập Project ID (unique, lowercase, no spaces): " PROJECT_ID
    read -p "Nhập Project Name (display name): " PROJECT_NAME
    
    echo "Đang tạo project..."
    gcloud projects create $PROJECT_ID --name="$PROJECT_NAME"
    
    if [ $? -eq 0 ]; then
        echo "✅ Project $PROJECT_ID đã được tạo!"
    else
        echo "❌ Không thể tạo project. Có thể PROJECT_ID đã tồn tại"
        echo "Vui lòng chọn PROJECT_ID khác"
        exit 1
    fi
else
    echo
    read -p "Nhập Project ID bạn muốn sử dụng: " PROJECT_ID
fi

# Set project
echo
echo "🎯 Cấu hình project active..."
gcloud config set project $PROJECT_ID

if [ $? -eq 0 ]; then
    echo "✅ Project $PROJECT_ID đã được set làm default"
else
    echo "❌ Không thể set project. Kiểm tra PROJECT_ID"
    exit 1
fi

echo
echo "📋 BƯỚC 4: ENABLE BILLING"
echo "========================="
echo

echo "⚠️  QUAN TRỌNG: Project cần có billing account để sử dụng Cloud Run"
echo "   - Google Cloud có free tier $300 credit cho tài khoản mới"
echo "   - Cloud Run có 2 triệu requests miễn phí mỗi tháng"
echo "   - Chi phí thực tế rất thấp cho usage thông thường"
echo

# Check billing
BILLING_ENABLED=$(gcloud billing projects describe $PROJECT_ID --format="value(billingEnabled)" 2>/dev/null)

if [ "$BILLING_ENABLED" = "True" ]; then
    echo "✅ Billing đã được enable cho project này"
else
    echo "❌ Billing chưa được enable"
    echo
    echo "Để enable billing:"
    echo "1. Truy cập: https://console.cloud.google.com/billing"
    echo "2. Chọn billing account hoặc tạo mới"
    echo "3. Link với project $PROJECT_ID"
    echo
    echo "Hoặc mở link này:"
    echo "https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT_ID"
    echo
    
    read -p "Nhấn Enter sau khi đã enable billing..."
fi

echo
echo "📋 BƯỚC 5: ENABLE APIs CẦN THIẾT"
echo "================================"
echo

echo "🔧 Đang enable APIs..."

# Enable required APIs
REQUIRED_APIS=(
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "containerregistry.googleapis.com"
    "artifactregistry.googleapis.com"
)

for api in "${REQUIRED_APIS[@]}"; do
    echo "   - Enabling $api..."
    gcloud services enable $api
    
    if [ $? -eq 0 ]; then
        echo "     ✅ $api enabled"
    else
        echo "     ❌ Failed to enable $api"
    fi
done

echo "✅ Tất cả APIs đã được enabled!"

echo
echo "📋 BƯỚC 6: CẤU HÌNH DOCKER"
echo "=========================="
echo

echo "🐳 Cấu hình Docker authentication..."
gcloud auth configure-docker

if [ $? -eq 0 ]; then
    echo "✅ Docker đã được cấu hình với Google Cloud"
else
    echo "❌ Lỗi cấu hình Docker"
fi

echo
echo "📋 BƯỚC 7: CẬP NHẬT DEPLOYMENT SCRIPTS"
echo "======================================"
echo

echo "📝 Cập nhật PROJECT_ID trong các file..."

# Update deploy.sh
if [ -f "deploy.sh" ]; then
    sed -i.bak "s/your-project-id/$PROJECT_ID/g" deploy.sh
    echo "✅ Updated deploy.sh với PROJECT_ID: $PROJECT_ID"
fi

# Update deploy-appengine.sh
if [ -f "deploy-appengine.sh" ]; then
    sed -i.bak "s/your-project-id/$PROJECT_ID/g" deploy-appengine.sh
    echo "✅ Updated deploy-appengine.sh với PROJECT_ID: $PROJECT_ID"
fi

echo
echo "📋 BƯỚC 8: BUILD VÀ DEPLOY"
echo "=========================="
echo

echo "🚀 Sẵn sàng deploy!"
echo
echo "Chọn phương pháp deploy:"
echo "1. Google Cloud Run (Khuyến nghị - Serverless, auto-scaling)"
echo "2. Google App Engine (Platform-as-a-Service)"
echo

read -p "Chọn (1/2): " deploy_choice

if [ "$deploy_choice" = "1" ]; then
    echo
    echo "🚀 Deploying to Cloud Run..."
    echo "Quá trình này có thể mất 5-10 phút..."
    echo
    
    # Run deploy script
    chmod +x deploy.sh
    ./deploy.sh
    
elif [ "$deploy_choice" = "2" ]; then
    echo
    echo "🚀 Deploying to App Engine..."
    echo "Quá trình này có thể mất 5-15 phút..."
    echo
    
    # Run deploy script
    chmod +x deploy-appengine.sh
    ./deploy-appengine.sh
    
else
    echo "❌ Lựa chọn không hợp lệ"
    exit 1
fi

echo
echo "🎉 SETUP HOÀN TẤT!"
echo "=================="
echo

echo "📋 Thông tin project:"
echo "   - Project ID: $PROJECT_ID"
echo "   - Region: asia-southeast1"
echo

if [ "$deploy_choice" = "1" ]; then
    echo "🌐 Service URL (Cloud Run):"
    SERVICE_URL=$(gcloud run services describe captcha-solver-api --platform managed --region asia-southeast1 --format 'value(status.url)' 2>/dev/null)
    if [ ! -z "$SERVICE_URL" ]; then
        echo "   $SERVICE_URL"
    else
        echo "   Kiểm tra trong Cloud Console: https://console.cloud.google.com/run"
    fi
else
    echo "🌐 App URL (App Engine):"
    echo "   https://$PROJECT_ID.appspot.com"
fi

echo
echo "📖 Test API:"
echo "   Health check:"
if [ "$deploy_choice" = "1" ]; then
    echo "   curl $SERVICE_URL/health"
else
    echo "   curl https://$PROJECT_ID.appspot.com/health"
fi

echo
echo "   Solve captcha:"
echo "   python client.py solve [API_URL] image.png"

echo
echo "🔧 Useful commands:"
echo "   - View logs: gcloud logs read"
echo "   - Update service: ./deploy.sh"
echo "   - Cloud Console: https://console.cloud.google.com"

echo
echo "💰 Chi phí:"
echo "   - Free tier: 2 triệu requests/tháng"
echo "   - Ước tính: $3-5/tháng cho usage thông thường"

echo
echo "🎉 CHÚC MỪNG! API CỦA BẠN ĐÃ ONLINE!"
