#!/bin/bash

echo "🔍 KIỂM TRA MÔI TRƯỜNG TRƯỚC KHI DEPLOY"
echo "========================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✅ $message${NC}"
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠️  $message${NC}"
    elif [ "$status" = "ERROR" ]; then
        echo -e "${RED}❌ $message${NC}"
    else
        echo -e "${BLUE}ℹ️  $message${NC}"
    fi
}

# Check functions
check_gcloud() {
    echo "🔍 Checking Google Cloud SDK..."
    
    if command -v gcloud &> /dev/null; then
        local version=$(gcloud --version | head -n1 | awk '{print $4}')
        print_status "OK" "Google Cloud SDK installed (version: $version)"
        
        # Check if authenticated
        local account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null)
        if [ ! -z "$account" ]; then
            print_status "OK" "Authenticated as: $account"
        else
            print_status "WARN" "Not authenticated - run 'gcloud auth login'"
        fi
        
        # Check current project
        local project=$(gcloud config get-value project 2>/dev/null)
        if [ ! -z "$project" ]; then
            print_status "OK" "Current project: $project"
        else
            print_status "WARN" "No project set - run 'gcloud config set project PROJECT_ID'"
        fi
        
        return 0
    else
        print_status "ERROR" "Google Cloud SDK not installed"
        echo "Install with: curl https://sdk.cloud.google.com | bash"
        return 1
    fi
}

check_docker() {
    echo
    echo "🔍 Checking Docker..."
    
    if command -v docker &> /dev/null; then
        local version=$(docker --version | awk '{print $3}' | sed 's/,//')
        print_status "OK" "Docker installed (version: $version)"
        
        # Check if Docker daemon is running
        if docker info &> /dev/null; then
            print_status "OK" "Docker daemon is running"
        else
            print_status "ERROR" "Docker daemon is not running"
            echo "Start Docker daemon and try again"
            return 1
        fi
        
        # Check Docker authentication with gcloud
        if docker-credential-gcloud list &> /dev/null; then
            print_status "OK" "Docker authenticated with Google Cloud"
        else
            print_status "WARN" "Docker not authenticated - run 'gcloud auth configure-docker'"
        fi
        
        return 0
    else
        print_status "ERROR" "Docker not installed"
        echo "Install with: curl -fsSL https://get.docker.com | sh"
        return 1
    fi
}

check_project_files() {
    echo
    echo "🔍 Checking project files..."
    
    local files=(
        "resolveCaptcha.py"
        "Dockerfile"
        "requirements.txt"
        "deploy.sh"
    )
    
    local missing_files=()
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            print_status "OK" "$file exists"
        else
            print_status "ERROR" "$file missing"
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        return 0
    else
        echo "Missing files: ${missing_files[*]}"
        return 1
    fi
}

check_project_config() {
    echo
    echo "🔍 Checking project configuration..."
    
    # Check if PROJECT_ID is set in deploy.sh
    if [ -f "deploy.sh" ]; then
        local project_id=$(grep 'PROJECT_ID=' deploy.sh | head -n1 | cut -d'"' -f2)
        if [ "$project_id" = "your-project-id" ]; then
            print_status "WARN" "PROJECT_ID not configured in deploy.sh"
            echo "Edit deploy.sh and set your actual PROJECT_ID"
        else
            print_status "OK" "PROJECT_ID configured: $project_id"
        fi
    fi
    
    # Check if image.png exists for testing
    if [ -f "image.png" ]; then
        print_status "OK" "Test image (image.png) available"
    else
        print_status "WARN" "No test image found - add image.png for testing"
    fi
}

check_apis() {
    echo
    echo "🔍 Checking Google Cloud APIs..."
    
    local project=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$project" ]; then
        print_status "WARN" "No project set - cannot check APIs"
        return 1
    fi
    
    local apis=(
        "cloudbuild.googleapis.com"
        "run.googleapis.com"
        "containerregistry.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        if gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
            print_status "OK" "$api enabled"
        else
            print_status "WARN" "$api not enabled"
            echo "Enable with: gcloud services enable $api"
        fi
    done
}

check_billing() {
    echo
    echo "🔍 Checking billing..."
    
    local project=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$project" ]; then
        print_status "WARN" "No project set - cannot check billing"
        return 1
    fi
    
    local billing_enabled=$(gcloud billing projects describe "$project" --format="value(billingEnabled)" 2>/dev/null)
    
    if [ "$billing_enabled" = "True" ]; then
        print_status "OK" "Billing enabled for project"
    else
        print_status "ERROR" "Billing not enabled"
        echo "Enable billing at: https://console.cloud.google.com/billing/linkedaccount?project=$project"
        return 1
    fi
}

check_network() {
    echo
    echo "🔍 Checking network connectivity..."
    
    # Test Google Cloud connectivity
    if curl -s --max-time 5 https://cloud.google.com > /dev/null; then
        print_status "OK" "Google Cloud reachable"
    else
        print_status "ERROR" "Cannot reach Google Cloud"
        return 1
    fi
    
    # Test Container Registry connectivity
    if curl -s --max-time 5 https://gcr.io > /dev/null; then
        print_status "OK" "Container Registry reachable"
    else
        print_status "ERROR" "Cannot reach Container Registry"
        return 1
    fi
}

run_quick_test() {
    echo
    echo "🔍 Running quick functionality test..."
    
    # Test Python imports
    if python3 -c "import cv2, numpy, PIL, flask" 2>/dev/null; then
        print_status "OK" "Python dependencies available"
    else
        print_status "WARN" "Some Python dependencies missing"
        echo "Run: pip install -r requirements.txt"
    fi
    
    # Test Docker build (dry run)
    if [ -f "Dockerfile" ]; then
        if docker build --dry-run -f Dockerfile . &> /dev/null; then
            print_status "OK" "Dockerfile syntax valid"
        else
            print_status "ERROR" "Dockerfile has syntax errors"
            return 1
        fi
    fi
}

provide_recommendations() {
    echo
    echo "📋 RECOMMENDATIONS"
    echo "=================="
    echo
    
    echo "Before deploying:"
    echo "1. ✅ Ensure all checks above are OK"
    echo "2. ✅ Test locally with: ./test-docker.sh"
    echo "3. ✅ Review deployment config in deploy.sh"
    echo "4. ✅ Have a test captcha image ready"
    echo
    
    echo "During deployment:"
    echo "1. 🕐 Be patient - first build takes 5-10 minutes"
    echo "2. 📊 Monitor progress in Cloud Console"
    echo "3. 📝 Note down the service URL for testing"
    echo
    
    echo "After deployment:"
    echo "1. 🔍 Test health endpoint: curl SERVICE_URL/health"
    echo "2. 🧪 Test solve endpoint with client.py"
    echo "3. 📊 Monitor logs and metrics"
    echo "4. 💰 Check billing usage periodically"
}

# Main execution
main() {
    local all_good=true
    
    check_gcloud || all_good=false
    check_docker || all_good=false
    check_project_files || all_good=false
    check_project_config
    check_apis
    check_billing || all_good=false
    check_network || all_good=false
    run_quick_test
    
    echo
    echo "📊 SUMMARY"
    echo "=========="
    
    if [ "$all_good" = true ]; then
        print_status "OK" "Environment ready for deployment! 🚀"
        echo
        echo "Next steps:"
        echo "1. Run: ./deploy.sh"
        echo "2. Wait for deployment to complete"
        echo "3. Test your API"
    else
        print_status "ERROR" "Please fix the issues above before deploying"
        echo
        echo "Common fixes:"
        echo "- Install missing tools"
        echo "- Authenticate with Google Cloud"
        echo "- Enable billing and APIs"
        echo "- Configure PROJECT_ID"
    fi
    
    provide_recommendations
}

# Run the main function
main "$@"
