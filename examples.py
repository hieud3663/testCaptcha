#!/usr/bin/env python3
"""
Ví dụ cụ thể sử dụng Captcha Solver API
"""

import requests
import base64
import json
import os

def example_basic_usage():
    """
    Ví dụ cơ bản sử dụng API
    """
    print("🔍 VÍ DỤ CƠ BẢN - GIẢI CAPTCHA")
    print("=" * 50)
    
    # URL API (thay đổi thành URL thực tế sau khi deploy)
    api_url = "https://your-service-name-xxxxx-as.a.run.app"  # Cloud Run
    # api_url = "https://your-project-id.appspot.com"          # App Engine
    # api_url = "http://localhost:8080"                        # Local testing
    
    # Đọc file ảnh và convert sang base64
    image_path = "captcha_example.png"  # Thay bằng path ảnh thực tế
    
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy file {image_path}")
        print("Vui lòng cung cấp file ảnh captcha để test")
        return
    
    # Convert image to base64
    with open(image_path, 'rb') as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
    
    print(f"📷 Đọc ảnh: {image_path}")
    print(f"📦 Kích thước base64: {len(base64_image)} ký tự")
    
    # Gửi request tới API
    try:
        response = requests.post(
            f"{api_url}/solve-captcha",
            json={"image": base64_image},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                coords = result['coordinates']
                print(f"✅ THÀNH CÔNG!")
                print(f"📍 Tọa độ tìm thấy: ({coords['x']}, {coords['y']})")
                print(f"📍 Tọa độ gốc: ({coords['raw_x']}, {coords['raw_y']})")
                print(f"💬 Message: {result['message']}")
            else:
                print(f"❌ THẤT BẠI: {result['error']}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")

def example_batch_processing():
    """
    Ví dụ xử lý batch (nhiều ảnh cùng lúc)
    """
    print("\n🔍 VÍ DỤ BATCH PROCESSING")
    print("=" * 50)
    
    api_url = "https://your-service-name-xxxxx-as.a.run.app"
    
    # List các file ảnh cần xử lý
    image_files = [
        "captcha1.png",
        "captcha2.png", 
        "captcha3.png"
    ]
    
    # Filter files that exist
    existing_files = [f for f in image_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ Không tìm thấy file ảnh nào để test batch")
        return
    
    print(f"📷 Tìm thấy {len(existing_files)} file ảnh")
    
    # Convert tất cả ảnh sang base64
    base64_images = []
    for image_path in existing_files:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            base64_images.append(base64_image)
    
    # Gửi batch request
    try:
        response = requests.post(
            f"{api_url}/solve-captcha-batch",
            json={"images": base64_images},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch completed!")
            print(f"📊 Tổng: {result['total']}")
            print(f"✅ Thành công: {result['success_count']}")
            print(f"❌ Thất bại: {result['failed_count']}")
            
            # Chi tiết từng kết quả
            for i, res in enumerate(result['results']):
                filename = existing_files[res['index']]
                if res['success']:
                    coords = res['coordinates']
                    print(f"  {i+1}. {filename}: ({coords['x']}, {coords['y']}) ✅")
                else:
                    print(f"  {i+1}. {filename}: {res['error']} ❌")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")

def example_web_integration():
    """
    Ví dụ tích hợp vào web application
    """
    print("\n🔍 VÍ DỤ TÍCH HỢP WEB")
    print("=" * 50)
    
    # Ví dụ HTML form
    html_example = '''
    <!-- HTML Form -->
    <form id="captcha-form" enctype="multipart/form-data">
        <input type="file" id="captcha-image" accept="image/*" required>
        <button type="submit">Giải Captcha</button>
    </form>
    
    <div id="result"></div>
    '''
    
    # Ví dụ JavaScript
    js_example = '''
    // JavaScript
    document.getElementById('captcha-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const fileInput = document.getElementById('captcha-image');
        const file = fileInput.files[0];
        
        if (!file) {
            alert('Vui lòng chọn file ảnh');
            return;
        }
        
        // Convert file to base64
        const reader = new FileReader();
        reader.onload = async (event) => {
            const base64 = event.target.result.split(',')[1]; // Remove data URL prefix
            
            try {
                const response = await fetch('https://your-api-url/solve-captcha', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: base64
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('result').innerHTML = 
                        `✅ Tọa độ: (${result.coordinates.x}, ${result.coordinates.y})`;
                } else {
                    document.getElementById('result').innerHTML = 
                        `❌ Lỗi: ${result.error}`;
                }
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    `❌ Network Error: ${error.message}`;
            }
        };
        
        reader.readAsDataURL(file);
    });
    '''
    
    print("HTML:")
    print(html_example)
    print("\nJavaScript:")
    print(js_example)

def example_mobile_integration():
    """
    Ví dụ tích hợp mobile (React Native)
    """
    print("\n🔍 VÍ DỤ TÍCH HỢP MOBILE")
    print("=" * 50)
    
    react_native_example = '''
    // React Native Example
    import React, { useState } from 'react';
    import { View, Button, Alert, Image } from 'react-native';
    import ImagePicker from 'react-native-image-picker';
    
    const CaptchaSolver = () => {
        const [result, setResult] = useState(null);
        
        const pickImage = () => {
            ImagePicker.showImagePicker({}, (response) => {
                if (response.data) {
                    solveCaptcha(response.data);
                }
            });
        };
        
        const solveCaptcha = async (base64Image) => {
            try {
                const response = await fetch('https://your-api-url/solve-captcha', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: base64Image
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    setResult(`Tọa độ: (${result.coordinates.x}, ${result.coordinates.y})`);
                    Alert.alert('Thành công', `Tọa độ: (${result.coordinates.x}, ${result.coordinates.y})`);
                } else {
                    Alert.alert('Lỗi', result.error);
                }
            } catch (error) {
                Alert.alert('Network Error', error.message);
            }
        };
        
        return (
            <View>
                <Button title="Chọn ảnh Captcha" onPress={pickImage} />
                {result && <Text>{result}</Text>}
            </View>
        );
    };
    '''
    
    print("React Native:")
    print(react_native_example)

def example_python_integration():
    """
    Ví dụ tích hợp vào Python application
    """
    print("\n🔍 VÍ DỤ TÍCH HỢP PYTHON")
    print("=" * 50)
    
    python_example = '''
    # Python Application Integration
    import requests
    import base64
    from typing import Optional, Tuple
    
    class CaptchaSolver:
        def __init__(self, api_url: str, api_key: Optional[str] = None):
            self.api_url = api_url.rstrip('/')
            self.api_key = api_key
            
        def solve_from_file(self, image_path: str) -> Optional[Tuple[int, int]]:
            """Giải captcha từ file ảnh"""
            try:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                
                return self.solve_from_base64(base64_image)
            except Exception as e:
                print(f"Error reading file: {e}")
                return None
        
        def solve_from_base64(self, base64_image: str) -> Optional[Tuple[int, int]]:
            """Giải captcha từ base64"""
            try:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["X-API-Key"] = self.api_key
                
                response = requests.post(
                    f"{self.api_url}/solve-captcha",
                    json={"image": base64_image},
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result['success']:
                        coords = result['coordinates']
                        return (coords['x'], coords['y'])
                
                return None
            except Exception as e:
                print(f"API Error: {e}")
                return None
    
    # Usage
    solver = CaptchaSolver("https://your-api-url")
    coordinates = solver.solve_from_file("captcha.png")
    
    if coordinates:
        x, y = coordinates
        print(f"Captcha solved: ({x}, {y})")
        # Use coordinates for automation...
    else:
        print("Failed to solve captcha")
    '''
    
    print("Python Class:")
    print(python_example)

def main():
    """
    Chạy tất cả ví dụ
    """
    print("🚀 CAPTCHA SOLVER API - VÍ DỤ SỬ DỤNG")
    print("=" * 60)
    
    # Chạy các ví dụ
    example_basic_usage()
    example_batch_processing()
    example_web_integration()
    example_mobile_integration()
    example_python_integration()
    
    print("\n🎉 HOÀN THÀNH TẤT CẢ VÍ DỤ!")
    print("=" * 60)
    print("\n📝 LƯU Ý:")
    print("- Thay thế 'https://your-api-url' bằng URL thực tế của service")
    print("- Chuẩn bị file ảnh captcha để test")
    print("- Kiểm tra network connection")
    print("- Monitor usage để tránh vượt quota")

if __name__ == "__main__":
    main()
