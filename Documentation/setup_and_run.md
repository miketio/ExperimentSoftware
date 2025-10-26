# Setup and Run Instructions

## 📦 Complete File List

You should have these files in your project:

### **Core System Files**
- `dual_thread_camera_stage_autofocus.py` - Original multi-threaded app (CLI only)
- `dual_thread_with_api.py` - **NEW** Multi-threaded app with REST API
- `stage_commands.py` - Stage command processor
- `xyzStageApp.py` - Stage application layer
- `smartactStage.py` - SmarAct hardware interface
- `xyzStageBase.py` - Abstract stage interface
- `mockStage.py` - Mock stage for testing
- `andorCameraApp.py` - Camera application layer
- `zylaCamera.py` - Zyla camera implementation
- `andorCameraBase.py` - Abstract camera interface

### **REST API Files (NEW)**
- `api_server.py` - **NEW** FastAPI server implementation
- `api_models.py` - **NEW** Pydantic models for API
- `test_api_client.py` - **NEW** Python client library + tests

### **Documentation**
- `README.md` - Project overview and documentation
- `API_QUICKSTART.md` - REST API quick start guide
- `SETUP_AND_RUN.md` - This file
- `requirements.txt` - Python dependencies

### **Example/Test Files**
- `exampleUsage.py` - Basic usage examples
- `run_tests.py` - Camera tests
- `try_different.py` - Alternative camera tests

### **Agent Framework (Future)**
- `basic_agent_create.py` - Agent example
- `multi_agent_coding_assistance.py` - Multi-agent demo
- `start_litellm.py` - LiteLLM proxy
- `test_proxy.py` - Proxy test
- `config.yaml` - LiteLLM config

---

## 🚀 Quick Start

### **Step 1: Install Dependencies**

```bash
# Activate your virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install new dependencies
pip install fastapi uvicorn[standard] pydantic requests
```

### **Step 2: Choose Your Mode**

#### **Option A: CLI Only (Original)**
```bash
python dual_thread_camera_stage_autofocus.py
```

#### **Option B: CLI + REST API (Recommended)**
```bash
python dual_thread_with_api.py
```

This starts:
- Thread 1: Camera live stream
- Thread 2: Stage control
- Thread 3: CLI input
- Thread 4: REST API server (http://localhost:5000)

### **Step 3: Verify API is Running**

Open browser: **http://localhost:5000/docs**

You should see interactive API documentation (Swagger UI)

### **Step 4: Test the API**

In a **separate terminal**:

```bash
# Quick health check
curl http://localhost:5000/health

# Or run full test suite
python test_api_client.py test
```

---

## 🎯 Usage Examples

### **Via CLI (Traditional)**

In the main application terminal:
```
>> pos
>> x 5000
>> autofocus
>> quit
```

### **Via REST API (New)**

In a separate terminal or script:

```bash
# Get current position
curl http://localhost:5000/status

# Move stage
curl -X POST http://localhost:5000/move/absolute \
  -H "Content-Type: application/json" \
  -d '{"axis": "x", "position": 5000}'

# Run autofocus
curl -X POST http://localhost:5000/autofocus \
  -H "Content-Type: application/json" \
  -d '{"axis": "x"}'
```

### **Via Python Client**

```python
from test_api_client import ExperimentAPIClient

client = ExperimentAPIClient()

# Check health
health = client.health_check()
print(health)

# Move stage
result = client.move_absolute("x", 5000)
print(f"Moved to {result['position']}nm")

# Run autofocus
result = client.run_autofocus(axis="x")
print(f"Best focus: {result['best_position']}nm")
```

---

## 🧪 Testing

### **Test 1: Basic API Functions**
```bash
python test_api_client.py test
```

Tests:
- Health check
- Get positions
- Move stage
- Camera info
- CLI command execution

### **Test 2: Autofocus**
```bash
python test_api_client.py autofocus
```

Runs a full autofocus scan via API.

### **Test 3: Interactive Demo**
```bash
python test_api_client.py demo
```

Interactive command-line demo.

---

## 📁 Recommended Project Structure

```
microscopy-control/
├── venv/                          # Virtual environment
│
├── Hardware Control
│   ├── dual_thread_with_api.py    # Main application (USE THIS)
│   ├── stage_commands.py          # Command processor
│   ├── xyzStageApp.py            # Stage app
│   ├── smartactStage.py          # SmarAct driver
│   ├── andorCameraApp.py         # Camera app
│   └── zylaCamera.py             # Zyla driver
│
├── REST API
│   ├── api_server.py             # FastAPI server
│   ├── api_models.py             # Pydantic models
│   └── test_api_client.py        # Client library
│
├── Base Classes
│   ├── xyzStageBase.py           # Stage interface
│   └── andorCameraBase.py        # Camera interface
│
├── Testing
│   ├── mockStage.py              # Mock hardware
│   ├── exampleUsage.py           # Examples
│   └── run_tests.py              # Tests
│
├── Agent Framework (Future)
│   ├── basic_agent_create.py
│   └── ...
│
└── Documentation
    ├── README.md
    ├── API_QUICKSTART.md
    └── SETUP_AND_RUN.md
```

---

## 🔧 Command Line Options

### **dual_thread_with_api.py**

```bash
# Default (API on port 5000)
python dual_thread_with_api.py

# Custom port
python dual_thread_with_api.py --api-port 8000

# Disable API (CLI only)
python dual_thread_with_api.py --no-api

# Help
python dual_thread_with_api.py --help
```

---

## 🐛 Troubleshooting

### **Problem: API not accessible**

**Solution:**
```bash
# 1. Check if app is running
curl http://localhost:5000/health

# 2. Check firewall
# Windows: Allow Python through firewall
# Linux: sudo ufw allow 5000

# 3. Try different port
python dual_thread_with_api.py --api-port 8000
```

### **Problem: Port already in use**

**Solution:**
```bash
# Find what's using port 5000
# Linux/Mac:
lsof -i :5000

# Windows:
netstat -ano | findstr :5000

# Use different port
python dual_thread_with_api.py --api-port 8000
```

### **Problem: Module not found**

**Solution:**
```bash
# Make sure you're in virtual environment
pip install -r requirements.txt

# Verify FastAPI installed
pip show fastapi
```

### **Problem: Camera/Stage not found**

**Solution:**
- Ensure hardware is connected and powered
- Verify SDKs are installed (Andor SDK3, SmarAct MCS)
- Check cables and USB connections
- Try with mock hardware first: `mockStage.py`

---

## 📊 Feature Comparison

| Feature | CLI Only | CLI + REST API |
|---------|----------|----------------|
| Manual control | ✅ | ✅ |
| Hardware access | ✅ | ✅ |
| Autofocus | ✅ | ✅ |
| Camera stream | ✅