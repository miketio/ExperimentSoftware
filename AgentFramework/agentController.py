# agent_controller.py
"""
AI Agent controller for experiment control via natural language.
Uses Microsoft Agent Framework with Google Gemini via LiteLLM.
"""
import asyncio
import json
from typing import Optional, Dict, Any
from openai import OpenAI

from Testing.test_api_client import ExperimentAPIClient
from AgentFramework.agentTools import ExperimentTools, create_tool_definitions
from AgentFramework.agentConfig import (
    LITELLM_BASE_URL,
    MODEL_NAME,
    API_KEY,
    EXPERIMENT_API_URL,
    AGENT_SYSTEM_PROMPT,
    Colors,
    EMOJI
)


class ExperimentAgent:
    """AI Agent for controlling microscopy experiments via natural language."""
    
    def __init__(self, api_url: str = EXPERIMENT_API_URL):
        """
        Initialize the experiment control agent.
        
        Args:
            api_url: URL of the experiment control API
        """
        self.api_client = ExperimentAPIClient(api_url)
        self.tools = ExperimentTools(self.api_client)
        
        # Initialize OpenAI client pointing to LiteLLM
        self.client = OpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=API_KEY
        )
        
        # Conversation history
        self.messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]
        
        # Pending confirmations
        self.pending_confirmation = None
        
        print(f"{EMOJI['success']} Agent initialized with model: {MODEL_NAME}")
        print(f"{EMOJI['success']} Connected to API: {api_url}")
    
    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool function with the given arguments.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            dict: Result from the tool
        """
        print(f"{Colors.OKCYAN}{EMOJI['tool']} Calling tool: {tool_name}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}   Arguments: {arguments}{Colors.ENDC}")
        
        # Map tool names to methods
        tool_map = {
            "get_current_position": self.tools.get_current_position,
            "move_axis_absolute": self.tools.move_axis_absolute,
            "move_axis_relative": self.tools.move_axis_relative,
            "run_autofocus": self.tools.run_autofocus,
            "get_autofocus_results": self.tools.get_autofocus_results,
            "get_camera_info": self.tools.get_camera_info,
            "check_system_health": self.tools.check_system_health,
        }
        
        if tool_name not in tool_map:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "message": f"Tool '{tool_name}' not found"
            }
        
        try:
            result = tool_map[tool_name](**arguments)
            
            # Check if confirmation is needed
            if not result.get("success") and result.get("needs_confirmation"):
                self.pending_confirmation = {
                    "tool_name": tool_name,
                    "arguments": arguments
                }
            
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Tool execution failed: {e}"
            }
    
    def handle_user_confirmation(self, user_response: str) -> Optional[Dict[str, Any]]:
        """
        Handle user confirmation for pending actions.
        
        Args:
            user_response: User's response to confirmation request
            
        Returns:
            dict: Result from executing the confirmed action, or None
        """
        if not self.pending_confirmation:
            return None
        
        response = user_response.lower().strip()
        
        if response in ['y', 'yes', 'confirm', 'ok']:
            # Execute the pending action with confirmation skip
            tool_name = self.pending_confirmation["tool_name"]
            arguments = self.pending_confirmation["arguments"]
            arguments["skip_confirmation"] = True
            
            result = self._call_tool(tool_name, arguments)
            self.pending_confirmation = None
            return result
        
        elif response in ['n', 'no', 'cancel', 'abort']:
            self.pending_confirmation = None
            return {
                "success": False,
                "cancelled": True,
                "message": "Action cancelled by user"
            }
        
        return None
    
    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.
        
        Args:
            user_message: Natural language message from user
            
        Returns:
            str: Agent's response
        """
        # Check if this is a confirmation response
        if self.pending_confirmation:
            confirmation_result = self.handle_user_confirmation(user_message)
            if confirmation_result is not None:
                if confirmation_result.get("cancelled"):
                    return f"{EMOJI['info']} Action cancelled."
                return f"{EMOJI['success']} {confirmation_result.get('message', 'Action completed')}"
        
        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})
        
        # Get tool definitions
        tools = create_tool_definitions()
        
        try:
            # Call LLM with tools
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.messages,
                tools=[{"type": "function", "function": tool} for tool in tools],
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            
            # Check if agent wants to use tools
            if assistant_message.tool_calls:
                # Add assistant message to history
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                # Execute tool calls
                tool_results = []
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    result = self._call_tool(function_name, function_args)
                    
                    # Add tool result to messages
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                    
                    tool_results.append(result)
                
                # Get final response from agent with tool results
                final_response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=self.messages
                )
                
                final_message = final_response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": final_message})
                
                return final_message
            
            else:
                # No tool calls, just return the response
                response_content = assistant_message.content or "I'm not sure how to help with that."
                self.messages.append({"role": "assistant", "content": response_content})
                return response_content
        
        except Exception as e:
            error_msg = f"{EMOJI['error']} Error communicating with agent: {e}"
            print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
            return error_msg
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]
        self.pending_confirmation = None
        print(f"{EMOJI['success']} Conversation reset")


# ========================================
# Standalone Chat Interface
# ========================================

async def chat_loop():
    """Interactive chat loop with the agent."""
    print("="*70)
    print(f"{EMOJI['agent']} EXPERIMENT CONTROL AI AGENT")
    print("="*70)
    print(f"\n{Colors.OKGREEN}Initializing agent...{Colors.ENDC}")
    
    try:
        agent = ExperimentAgent()
    except Exception as e:
        print(f"\n{Colors.FAIL}{EMOJI['error']} Failed to initialize agent: {e}{Colors.ENDC}")
        print(f"\n{Colors.WARNING}Make sure:{Colors.ENDC}")
        print("  1. LiteLLM proxy is running: python start_litellm.py")
        print("  2. Experiment control API is running: python dual_thread_with_api.py")
        print("  3. .env file has GEMINI_API_KEY set")
        return
    
    print(f"\n{Colors.OKGREEN}✅ Agent ready!{Colors.ENDC}\n")
    print(f"{Colors.OKCYAN}Commands:{Colors.ENDC}")
    print("  • Type your request in natural language")
    print("  • 'reset' - Clear conversation history")
    print("  • 'quit' or 'exit' - Exit agent")
    print(f"\n{Colors.OKCYAN}Examples:{Colors.ENDC}")
    print('  • "What\'s the current position?"')
    print('  • "Move X to 8000"')
    print('  • "Run autofocus on X axis"')
    print('  • "Move to X=5000, Y=3000, then autofocus"')
    print("="*70 + "\n")
    
    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.BOLD}{EMOJI['user']} You: {Colors.ENDC}").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(f"\n{EMOJI['info']} Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                agent.reset_conversation()
                continue
            
            # Get agent response
            print(f"{Colors.OKBLUE}{EMOJI['thinking']} Agent: {Colors.ENDC}", end="", flush=True)
            response = await agent.chat(user_input)
            print(f"\r{Colors.OKBLUE}{EMOJI['agent']} Agent: {Colors.ENDC}{response}\n")
        
        except KeyboardInterrupt:
            print(f"\n\n{EMOJI['info']} Interrupted. Goodbye!")
            break
        except EOFError:
            print(f"\n\n{EMOJI['info']} EOF received. Goodbye!")
            break
        except Exception as e:
            print(f"\n{Colors.FAIL}{EMOJI['error']} Error: {e}{Colors.ENDC}\n")


async def main():
    """Main entry point."""
    await chat_loop()


if __name__ == "__main__":
    asyncio.run(main())