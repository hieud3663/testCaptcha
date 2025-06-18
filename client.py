import requests
import base64
import json
import time

class CaptchaSolverClient:
    def __init__(self, api_url):
        """
        Initialize client với API URL
        
        Args:
            api_url: URL của API (ví dụ: https://your-service.a.run.app)
        """
        self.api_url = api_url.rstrip('/')
        
    def health_check(self):
        """
        Kiểm tra health của API
        """
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def solve_captcha(self, image_path=None, base64_image=None):
        """
        Giải captcha từ file ảnh hoặc base64
        
        Args:
            image_path: Đường dẫn tới file ảnh
            base64_image: Chuỗi base64 của ảnh
            
        Returns:
            dict: Kết quả từ API
        """
        try:
            if image_path:
                # Đọc file và convert sang base64
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
            
            if not base64_image:
                return {"error": "Cần cung cấp image_path hoặc base64_image"}
            
            # Gửi request tới API
            payload = {
                "image": base64_image
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/solve-captcha", 
                json=payload, 
                headers=headers,
                timeout=30
            )
            end_time = time.time()
            
            result = response.json()
            result['processing_time'] = round(end_time - start_time, 2)
            result['status_code'] = response.status_code
            
            return result
            
        except requests.exceptions.RequestException as e:
            return {"error": f"Request error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def solve_captcha_batch(self, image_paths=None, base64_images=None):
        """
        Giải nhiều captcha cùng lúc
        
        Args:
            image_paths: List các đường dẫn tới file ảnh
            base64_images: List các chuỗi base64
            
        Returns:
            dict: Kết quả từ API
        """
        try:
            images = []
            
            if image_paths:
                for image_path in image_paths:
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                        base64_image = base64.b64encode(image_data).decode('utf-8')
                        images.append(base64_image)
            elif base64_images:
                images = base64_images
            else:
                return {"error": "Cần cung cấp image_paths hoặc base64_images"}
            
            # Gửi request tới API
            payload = {
                "images": images
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/solve-captcha-batch", 
                json=payload, 
                headers=headers,
                timeout=60
            )
            end_time = time.time()
            
            result = response.json()
            result['processing_time'] = round(end_time - start_time, 2)
            result['status_code'] = response.status_code
            
            return result
            
        except requests.exceptions.RequestException as e:
            return {"error": f"Request error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

def demo_usage():
    """
    Demo cách sử dụng client
    """
    # Khởi tạo client với URL API
    # Thay thế URL này bằng URL thực tế của service trên Google Cloud
    api_url = "https://your-service-name-hash-as.a.run.app"  # Cloud Run URL
    # hoặc api_url = "https://your-project-id.appspot.com"    # App Engine URL
    # hoặc api_url = "http://localhost:8080"                   # Local testing
    
    client = CaptchaSolverClient(api_url)
    
    print("🔍 Demo Captcha Solver Client")
    print(f"🌐 API URL: {api_url}")
    print()
    
    # 1. Health check
    print("1️⃣ Health Check...")
    health = client.health_check()
    print(f"Health: {json.dumps(health, indent=2, ensure_ascii=False)}")
    print()
    
    # 2. Giải single captcha
    print("2️⃣ Solve Single Captcha...")
    if os.path.exists("image.png"):
        result = client.solve_captcha(image_path="image.png")
        print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        if result.get('success'):
            coords = result['coordinates']
            print(f"✅ Tìm thấy tọa độ: ({coords['x']}, {coords['y']})")
            print(f"⏱️ Thời gian xử lý: {result['processing_time']}s")
        else:
            print(f"❌ Thất bại: {result.get('error', 'Unknown error')}")
    else:
        print("❌ Không tìm thấy file image.png để test")
    print()
    
    # 3. Giải batch captcha (nếu có nhiều ảnh)
    print("3️⃣ Solve Batch Captcha...")
    image_files = ["image.png", "image2.png", "image3.png"]  # Thay đổi theo file thực tế
    existing_files = [f for f in image_files if os.path.exists(f)]
    
    if existing_files:
        print(f"Tìm thấy {len(existing_files)} file: {existing_files}")
        batch_result = client.solve_captcha_batch(image_paths=existing_files)
        print(f"Batch Result: {json.dumps(batch_result, indent=2, ensure_ascii=False)}")
        
        if batch_result.get('success'):
            print(f"✅ Xử lý {batch_result['success_count']}/{batch_result['total']} ảnh thành công")
            print(f"⏱️ Thời gian xử lý: {batch_result['processing_time']}s")
        else:
            print(f"❌ Batch thất bại: {batch_result.get('error', 'Unknown error')}")
    else:
        print("❌ Không tìm thấy file ảnh nào để test batch")

def test_with_base64():
    """
    Test với base64 image cố định
    """
    # Tiny 1x1 pixel PNG (for testing only)
    tiny_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    api_url = "http://localhost:8080"  # Local testing
    client = CaptchaSolverClient(api_url)
    
    print("🧪 Test với Base64 Image")
    result = client.solve_captcha(base64_image=tiny_png_base64)
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    import os
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "demo":
            demo_usage()
        elif command == "test":
            test_with_base64()
        elif command == "health":
            api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080"
            client = CaptchaSolverClient(api_url)
            health = client.health_check()
            print(json.dumps(health, indent=2, ensure_ascii=False))
        elif command == "solve":
            api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080"
            image_path = sys.argv[3] if len(sys.argv) > 3 else "image.png"
            
            client = CaptchaSolverClient(api_url)
            result = client.solve_captcha(image_path=image_path)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("❌ Lệnh không hợp lệ!")
            print("Sử dụng:")
            print("  python client.py demo                           - Demo đầy đủ")
            print("  python client.py test                           - Test với base64")
            print("  python client.py health [api_url]               - Health check")
            print("  python client.py solve [api_url] [image_path]   - Solve captcha")
    else:
        demo_usage()
