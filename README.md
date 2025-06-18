# Captcha Solver API - Google Cloud Deployment

Dự án API giải captcha puzzle sử dụng computer vision và machine learning, được triển khai trên Google Cloud.

## 🚀 Triển khai nhanh

### Phương pháp 1: Google Cloud Run (Khuyến nghị)

```bash
# 1. Cài đặt Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# 2. Đăng nhập và cấu hình project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 3. Chỉnh sửa PROJECT_ID trong deploy.sh
nano deploy.sh

# 4. Chạy script deploy
chmod +x deploy.sh
./deploy.sh
```

### Phương pháp 2: Google App Engine

```bash
# 1. Chỉnh sửa PROJECT_ID trong deploy-appengine.sh
nano deploy-appengine.sh

# 2. Deploy lên App Engine
chmod +x deploy-appengine.sh
./deploy-appengine.sh
```

## 🧪 Test trước khi deploy

```bash
# Test Docker locally
chmod +x test-docker.sh
./test-docker.sh

# Hoặc build manual
docker build -t captcha-solver .
docker run -p 8080:8080 captcha-solver
```

## 📖 Cách sử dụng API

### Health Check
```bash
curl https://your-service-url/health
```

### Giải single captcha
```bash
curl -X POST https://your-service-url/solve-captcha \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_here"
  }'
```

### Giải batch captcha
```bash
curl -X POST https://your-service-url/solve-captcha-batch \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["base64_image1", "base64_image2"]
  }'
```

## 🐍 Sử dụng Python Client

```python
from client import CaptchaSolverClient

# Khởi tạo client
client = CaptchaSolverClient("https://your-service-url")

# Health check
health = client.health_check()
print(health)

# Giải captcha
result = client.solve_captcha(image_path="captcha.png")
if result['success']:
    x, y = result['coordinates']['x'], result['coordinates']['y']
    print(f"Tọa độ: ({x}, {y})")
```

## 📁 Cấu trúc files

```
testCaptcha/
├── resolveCaptcha.py      # Main API code
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── app.yaml              # App Engine config
├── deploy.sh             # Cloud Run deployment
├── deploy-appengine.sh   # App Engine deployment
├── test-docker.sh        # Local Docker testing
├── client.py             # Python client library
└── README.md             # Hướng dẫn này
```

## ⚙️ Cấu hình

### Environment Variables
- `FLASK_ENV`: production
- `PORT`: 8080 (mặc định)

### Resource Requirements
- **Memory**: 2GB
- **CPU**: 2 cores
- **Timeout**: 300 seconds
- **Max Instances**: 10

## 🔧 Troubleshooting

### Build Docker thất bại
```bash
# Cài đặt dependencies trước
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### Deploy thất bại
```bash
# Enable APIs
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com
```

### Memory issues
```bash
# Tăng memory limit
gcloud run deploy your-service \
    --memory 4Gi \
    --cpu 2
```

## 💰 Chi phí ước tính

### Google Cloud Run
- **Free tier**: 2 triệu requests/tháng
- **Sau free tier**: ~$0.40 mỗi triệu requests
- **Memory**: ~$0.0000025/GB-second

### Google App Engine
- **Standard Environment**: Tương tự Cloud Run
- **Flexible Environment**: Cao hơn (~$0.05/instance-hour)

## 🔒 Bảo mật

### Production recommendations:
1. **Enable authentication**:
   ```bash
   gcloud run deploy your-service --no-allow-unauthenticated
   ```

2. **API Key protection**:
   ```python
   # Thêm vào resolveCaptcha.py
   @app.before_request
   def check_api_key():
       if request.endpoint != 'health_check':
           api_key = request.headers.get('X-API-Key')
           if api_key != os.environ.get('API_KEY'):
               return jsonify({'error': 'Invalid API key'}), 401
   ```

3. **Rate limiting**:
   ```python
   from flask_limiter import Limiter
   
   limiter = Limiter(
       app,
       key_func=lambda: request.remote_addr,
       default_limits=["100 per hour"]
   )
   ```

## 📊 Monitoring

### Logs
```bash
# Cloud Run logs
gcloud logs read --service=captcha-solver-api

# App Engine logs
gcloud app logs tail -s default
```

### Metrics
- Cloud Console > Cloud Run/App Engine
- Monitor latency, errors, traffic

## 🔄 Updates

### Update service:
```bash
# Rebuild và redeploy
./deploy.sh
```

### Rollback:
```bash
# Cloud Run
gcloud run revisions list --service=captcha-solver-api
gcloud run services update-traffic captcha-solver-api \
    --to-revisions=REVISION_NAME=100

# App Engine
gcloud app versions list
gcloud app services set-traffic default \
    --splits=VERSION_ID=1.0
```

## 🆘 Support

Nếu gặp vấn đề:

1. Kiểm tra logs: `gcloud logs read`
2. Test local: `./test-docker.sh`
3. Kiểm tra quotas: Cloud Console > IAM & Admin > Quotas
4. Health check: `curl https://your-service/health`

## 📝 License

MIT License
