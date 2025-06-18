#!/bin/bash

# Script để build và test Docker image locally trước khi deploy

IMAGE_NAME="captcha-solver-api"
CONTAINER_NAME="captcha-api-test"
PORT=8080

echo "🐳 Build và Test Docker Image Local"
echo

# 1. Build Docker image
echo "1️⃣ Build Docker image..."
docker build -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo "❌ Build Docker image thất bại!"
    exit 1
fi

# 2. Stop và remove container cũ (nếu có)
echo "2️⃣ Dọn dẹp container cũ..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# 3. Run container
echo "3️⃣ Chạy container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8080 \
    $IMAGE_NAME

if [ $? -ne 0 ]; then
    echo "❌ Chạy container thất bại!"
    exit 1
fi

# 4. Đợi container start up
echo "4️⃣ Đợi container khởi động..."
sleep 10

# 5. Test health check
echo "5️⃣ Test health check..."
HEALTH_RESPONSE=$(curl -s http://localhost:$PORT/health)
echo "Health check response: $HEALTH_RESPONSE"

if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
    echo "✅ Health check PASSED!"
else
    echo "❌ Health check FAILED!"
    echo "Container logs:"
    docker logs $CONTAINER_NAME
    exit 1
fi

# 6. Test với ảnh mẫu (nếu có)
if [ -f "image.png" ]; then
    echo "6️⃣ Test với ảnh mẫu..."
    
    # Convert image to base64
    BASE64_IMAGE=$(base64 -w 0 image.png)
    
    # Test API
    TEST_RESPONSE=$(curl -s -X POST http://localhost:$PORT/solve-captcha \
        -H "Content-Type: application/json" \
        -d "{\"image\": \"$BASE64_IMAGE\"}")
    
    echo "API test response:"
    echo $TEST_RESPONSE | python3 -m json.tool 2>/dev/null || echo $TEST_RESPONSE
else
    echo "6️⃣ Không tìm thấy image.png để test"
fi

echo
echo "🎉 DOCKER TEST HOÀN THÀNH!"
echo "🌐 API đang chạy tại: http://localhost:$PORT"
echo
echo "📋 Các lệnh hữu ích:"
echo "  - Xem logs: docker logs $CONTAINER_NAME"
echo "  - Stop container: docker stop $CONTAINER_NAME"
echo "  - Remove container: docker rm $CONTAINER_NAME"
echo "  - Remove image: docker rmi $IMAGE_NAME"
echo
echo "🚀 Nếu test OK, có thể deploy lên Google Cloud bằng:"
echo "  ./deploy.sh"
