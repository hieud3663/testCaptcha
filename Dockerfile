# Sử dụng Python 3.12 slim image
FROM python:3.12-slim

# Cài đặt system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Tạo working directory
WORKDIR /app

# Copy requirements trước để cache layer
COPY requirements.txt .

# Cài đặt setuptools và wheel trước
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Cài đặt Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Tạo thư mục temp để xử lý file
RUN mkdir -p /tmp/captcha_temp

# Expose port
EXPOSE 8080

# Set environment variables
ENV FLASK_APP=resolveCaptcha.py
ENV FLASK_ENV=production
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run command
CMD ["python", "resolveCaptcha.py", "api", "0.0.0.0", "8080"]
