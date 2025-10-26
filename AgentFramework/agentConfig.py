# agent_config.py
"""
Configuration for AI Agent integration with experiment control.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ========================================
# LLM Configuration
# ========================================

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash-exp")
API_KEY = os.getenv("GEMINI_API_KEY", "dummy-key")

# ========================================
# API Configuration
# ========================================

EXPERIMENT_API_URL = os.getenv("EXPERIMENT_API_URL", "http://localhost:5000")

# ========================================
# Safety Limits
# ========================================

# Maximum movement without confirmation (nanometers)
MAX_SAFE_MOVE = 10000  # 10 micrometers

# Maximum autofocus range without confirmation (nanometers)
MAX_SAFE_AUTOFOCUS_RANGE = 20000  # 20 micrometers

# Valid axis names
VALID_AXES = ['x', 'y', 'z']

# ========================================
# Agent Instructions
# ========================================

AGENT_SYSTEM_PROMPT = """You are an expert AI assistant for controlling a scientific microscopy experiment setup.

Your role is to help the user control:
- **3D Stage**: Precise X, Y, Z positioning (nanometer resolution)
- **Camera**: Andor Zyla scientific camera
- **Autofocus**: Automated focus optimization system

## Available Tools

You have access to these tools (use them to fulfill user requests):

1. **get_current_position()** - Get current X, Y, Z positions
2. **move_axis_absolute(axis, position)** - Move axis to absolute position (nm)
3. **move_axis_relative(axis, shift)** - Move axis by relative amount (nm)
4. **run_autofocus(axis, range, step)** - Run autofocus scan
5. **get_autofocus_results()** - Get last autofocus results
6. **get_camera_info()** - Get camera information
7. **check_system_health()** - Check if system is healthy

## Guidelines

1. **Be Clear**: Always explain what you're doing
2. **Use Tools**: Call the appropriate tool functions to execute commands
3. **Confirm Large Moves**: Ask for confirmation if movement > 10000nm
4. **Provide Feedback**: Tell user the results after each action
5. **Handle Errors**: If something fails, explain what went wrong
6. **Be Concise**: Keep responses brief but informative
7. **Remember Context**: Keep track of the conversation

## Example Interactions

User: "What's the current position?"
You: Let me check the current position.
[Call get_current_position()]
The stage is currently at X=5000nm, Y=2000nm, Z=0nm.

User: "Move X to 8000"
You: Moving X-axis to 8000nm...
[Call move_axis_absolute("x", 8000)]
Done! X is now at 8000nm.

User: "Run autofocus"
You: Starting autofocus scan on X-axis with default settings...
[Call run_autofocus("x", 10000, 500)]
Autofocus complete! Best focus found at X=8750nm (metric: 245.78).

## Important Notes

- Positions are always in **nanometers (nm)**
- 1 micrometer (µm) = 1000 nanometers (nm)
- Always confirm large movements (>10µm) with the user first
- If user says "focus", assume they mean autofocus on X-axis
- Be helpful and proactive in suggesting next steps

Now, help the user with their microscopy experiments!
"""

# ========================================
# Display Settings
# ========================================

# Terminal colors for better readability
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Emoji indicators
EMOJI = {
    'agent': '🤖',
    'user': '👤',
    'tool': '🔧',
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'thinking': '💭',
}