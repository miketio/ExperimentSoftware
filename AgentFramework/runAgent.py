#!/usr/bin/env python3
# run_agent.py
"""
Simple launcher script for the Experiment Control AI Agent.
"""
import sys
import asyncio
from AgentFramework.agentController import chat_loop


def print_banner():
    """Print startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║         🤖  EXPERIMENT CONTROL AI AGENT  🔬                   ║
    ║                                                               ║
    ║     Natural Language Interface for Microscopy Control        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_prerequisites():
    """Check if required services are accessible."""
    import requests
    
    print("Checking prerequisites...\n")
    
    # Check LiteLLM proxy
    try:
        response = requests.get("http://localhost:4000/health", timeout=2)
        print("✅ LiteLLM proxy is running (port 4000)")
    except:
        print("❌ LiteLLM proxy NOT running")
        print("   → Start it: python start_litellm.py")
        return False
    
    # Check Experiment Control API
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        print("✅ Experiment Control API is running (port 5000)")
    except:
        print("❌ Experiment Control API NOT running")
        print("   → Start it: python dual_thread_with_api.py")
        return False
    
    print("\n✅ All prerequisites satisfied!\n")
    return True


def main():
    """Main entry point."""
    print_banner()
    
    # Check if services are running
    if not check_prerequisites():
        print("\n⚠️  Please start the required services first.\n")
        sys.exit(1)
    
    # Start agent chat loop
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()