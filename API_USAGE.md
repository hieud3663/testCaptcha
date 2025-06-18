# Captcha Puzzle Solver API

## Cách sử dụng

### 1. Chạy API Server
```bash
python resolveCaptcha.py api
```

Hoặc chỉ định host và port:
```bash
python resolveCaptcha.py api 0.0.0.0 8000
```

### 2. Test với ảnh local
```bash
python resolveCaptcha.py test image.png
```

### 3. Xử lý trực tiếp
```bash
python resolveCaptcha.py solve image.png
```

## API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```

### Xử lý 1 captcha
```bash
curl -X POST http://localhost:5000/solve-captcha \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_string_here"
  }'
```

### Xử lý nhiều captcha
```bash
curl -X POST http://localhost:5000/solve-captcha-batch \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["base64_1", "base64_2", "base64_3"]
  }'
```

## Response Format

### Success Response
```json
{
  "success": true,
  "coordinates": {
    "x": 257,
    "y": 26,
    "raw_x": 275,
    "raw_y": 26
  },
  "message": "Tìm thấy vị trí puzzle gap thành công"
}
```

### Error Response
```json
{
  "success": false,
  "error": "Không tìm thấy vị trí puzzle gap",
  "coordinates": null
}
```

## Python Client Example

```python
import requests
import base64

# Đọc ảnh và convert sang base64
with open('captcha.png', 'rb') as f:
    image_data = f.read()
    base64_string = base64.b64encode(image_data).decode('utf-8')

# Gửi request
response = requests.post('http://localhost:5000/solve-captcha', 
                        json={'image': base64_string})

if response.status_code == 200:
    result = response.json()
    if result['success']:
        x, y = result['coordinates']['x'], result['coordinates']['y']
        print(f"Tọa độ puzzle gap: ({x}, {y})")
    else:
        print(f"Lỗi: {result['error']}")
else:
    print(f"HTTP Error: {response.status_code}")
```

## JavaScript Client Example

```javascript
// Đọc file và convert sang base64
const fileInput = document.getElementById('fileInput');
const file = fileInput.files[0];

const reader = new FileReader();
reader.onload = async function(e) {
    const base64String = e.target.result.split(',')[1]; // Loại bỏ header
    
    try {
        const response = await fetch('http://localhost:5000/solve-captcha', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: base64String
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            const {x, y} = result.coordinates;
            console.log(`Tọa độ puzzle gap: (${x}, ${y})`);
        } else {
            console.error('Lỗi:', result.error);
        }
    } catch (error) {
        console.error('Network error:', error);
    }
};

reader.readAsDataURL(file);
```

## Features

- ✅ Thuật toán siêu chính xác với 4 phương pháp kết hợp
- ✅ API REST đơn giản, dễ sử dụng
- ✅ Xử lý batch (nhiều ảnh cùng lúc)
- ✅ Error handling toàn diện
- ✅ Support multiple image formats
- ✅ Automatic cleanup temp files
- ✅ Health check endpoint
- ✅ Detailed logging

## Requirements

```bash
pip install flask pillow numpy opencv-python-headless scikit-image scipy
```
