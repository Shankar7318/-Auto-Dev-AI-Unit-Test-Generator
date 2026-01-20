# 🚀 Auto-Dev: AI Unit Test Generator

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

Automatically generate **comprehensive unit tests** for your Python code using **AI-powered analysis and test generation**.

---

## 📖 Table of Contents
- ✨ Features  
- 🚀 Quick Start  
- 🏗️ Architecture  
- 🔧 Installation  
- 📊 Usage Guide  
- ⚙️ Configuration  
- 🧪 Test Generation  
- 🛠️ Development  

---

## ✨ Features

### 🎯 Core Capabilities
- 📁 **Multi-format Code Input** – Upload `.py` files or write code in the built-in editor  
- 🤖 **AI-Powered Analysis** – Automatically parses code and identifies testable components  
- ⚡ **Smart Test Generation** – AI-driven or template-based unit test generation  
- 📈 **Multiple Iterations** – Iteratively improve tests with configurable limits  

### 🎨 User Interface
- 🖥️ **Modern Web Interface** – Clean Streamlit UI with real-time previews  
- 📊 **Visual Analytics** – Code statistics and complexity insights  
- 🎛️ **Configurable Settings** – Control test style, coverage, and AI models  
- 📱 **Responsive Design** – Works on desktop and tablets  

### 🔧 Technical Features
- 🧠 **Multiple AI Models** – CodeLlama, GPT-4, Claude, or custom LLMs  
- 📚 **Test Templates** – Basic, Advanced, and Comprehensive  
- 🔄 **Version Control** – Track and compare test versions  
- 💾 **Export Options** – Download `.py`, copy to clipboard, or save to project  

---

## 🚀 Quick Start

### Prerequisites
- Python **3.8+**
- `pip`
- **4GB RAM minimum** (8GB+ recommended for LLM features)

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/auto-dev-unit-test-generator.git
cd auto-dev-unit-test-generator

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run main.py

# Enable AI Features
pip install torch transformers accelerate bitsandbytes
```
### 🏗️ Architecture
   # System Overview
   ```
   ┌─────────────────────────────────────────────┐
│           Streamlit Web Interface           │
├─────────────────────────────────────────────┤
│ Upload │ Analyze │ Generate │ Export        │
├─────────────────────────────────────────────┤
│           Core Processing Engine            │
├─────────────────────────────────────────────┤
│ Code Parser │ Test Generator │ AI Models    │
└─────────────────────────────────────────────┘

```
## Directory Structure
```
auto-dev-unit-test-generator/
├── main.py
├── requirements.txt
├── README.md
├── generated_tests/
├── coverage_reports/
└── src/
    ├── models/
    │   └── local_llm.py
    ├── config/
    │   └── settings.py
    └── utils/
        └── code_parser.py
```
# 📊 Usage Guide

- Workflow

- Upload .py file or paste code

- Click Analyze Code Structure

- Choose test generation strategy

- Review, run, and export tests
# Example Source Code
```
class Calculator:
    def add(self, a, b):
        return a + b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

```
# Generated Tests
```
import pytest

class TestCalculator:
    def test_add(self):
        assert Calculator().add(2, 3) == 5

    def test_divide_by_zero(self):
        with pytest.raises(ValueError):
            Calculator().divide(10, 0)
```
## ⚙️ Configuration
```
class Settings:
    MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
    MAX_TOKENS = 2048
    TEMPERATURE = 0.1
    TARGET_COVERAGE = 95
    MAX_ITERATIONS = 5
```
# Environment Variables

```
export MODEL_NAME="codellama/CodeLlama-13b-Instruct-hf"
export MAX_TOKENS=4096
export TEMPERATURE=0.2
```

# Test Generation Styles
```
| Feature           | Basic | Advanced | Comprehensive | AI |
| ----------------- | ----- | -------- | ------------- | -- |
| Function Tests    | ✅     | ✅        | ✅             | ✅  |
| Fixtures          | ⚠️    | ✅        | ✅             | ✅  |
| Mocking           | ❌     | ✅        | ✅             | ✅  |
| Integration Tests | ❌     | ❌        | ✅             | ✅  |
| AI Optimized      | ❌     | ❌        | ❌             | ✅  |
```

# 🛠️ Development
 - Dev Setup
 ```
git clone --branch develop https://github.com/yourusername/auto-dev-unit-test-generator.git
pip install -r requirements-dev.txt
pre-commit install
```
## Run Tests
# pytest
pytest --cov=src --cov-report=html

# Debug Mode
streamlit run main.py --logger.level=debug

## 📦 Usage Instructions

This project supports **local execution, Docker-based development, GPU acceleration, production deployment, and Kubernetes orchestration**.

---

## 1️⃣ Basic Docker Usage

### Build the Docker Image
```bash
docker build -t auto-dev .
Run the Container
docker run -p 8501:8501 \
  -v $(pwd)/generated_tests:/app/generated_tests \
  auto-dev
Using Docker Compose
docker-compose up -d
```
 ## 2️⃣ Development with Docker
- Optimized for rapid development and debugging.

### Start Development Environment
```
make dev
View Logs
docker-compose logs -f auto-dev
```
### Run Tests Inside the Container
- docker-compose exec auto-dev pytest tests/
#### Open Interactive Shell
- docker-compose exec auto-dev bash
## 4️⃣ Production Deployment
Recommended for stable, long-running production workloads.
```
Build Production Image
docker build -t auto-dev:prod .
Run Production Container
docker run -d \
  -p 8501:8501 \
  -v /data/auto-dev/tests:/app/generated_tests \
  -v /data/auto-dev/coverage:/app/coverage_reports \
  -e MODEL_NAME=codellama/CodeLlama-13b-Instruct-hf \
  --name auto-dev-prod \
  auto-dev:prod
  ```
Persistent volumes ensure test artifacts and coverage reports survive container restarts.

## 5️⃣ Kubernetes Deployment
Auto-Dev is Kubernetes-ready for scalable cloud-native deployments.
```
Apply Kubernetes Manifests
kubectl apply -f kubernetes/
Check Deployment Status
kubectl get pods -l app=auto-dev
View Logs
kubectl logs -f deployment/auto-dev
Access the Application
kubectl port-forward service/auto-dev-service 8501:8501 ```
