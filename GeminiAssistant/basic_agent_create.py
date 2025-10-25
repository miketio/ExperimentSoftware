import asyncio
from random import randint
import os

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIResponsesClient


# Tool functions
def get_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city or location to get weather for
        
    Returns:
        Weather description string
    """
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    temp = randint(10, 30)
    condition = conditions[randint(0, 3)]
    return f"The weather in {location} is {condition} with a high of {temp}°C."


def get_temperature(location: str, unit: str = "celsius") -> str:
    """Get temperature for a location.
    
    Args:
        location: The city or location
        unit: Temperature unit ('celsius' or 'fahrenheit')
        
    Returns:
        Temperature string
    """
    temp_c = randint(10, 30)
    if unit.lower() == "fahrenheit":
        temp_f = (temp_c * 9/5) + 32
        return f"Temperature in {location}: {temp_f:.1f}°F"
    return f"Temperature in {location}: {temp_c}°C"


# Create OpenAI client pointing to LiteLLM proxy
chat_client = OpenAIResponsesClient(
    base_url="http://localhost:4000/v1",
    model=os.getenv("MODEL_NAME", "gemini-2.0-flash"),
    model_id=os.getenv("MODEL_NAME", "gemini-2.0-flash"),
    api_key="dummy-key"
)

# Create agent with multiple tools
agent = ChatAgent(
    chat_client=chat_client,
    instructions="""You are a helpful weather assistant. 
    Use the available tools to get weather information when asked.
    Always provide friendly and informative responses.""",
    tools=[get_weather, get_temperature],
)


async def chat_loop():
    """Interactive chat loop with the agent."""
    print("="*70)
    print("🌤️  Weather Agent - Chat Interface")
    print("="*70)
    print("Ask about weather or temperature in any location!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            # Run agent
            print("Agent: ", end="", flush=True)
            result = await agent.run(user_input)
            
            # Extract final response text
            # The agent automatically handles tool calls internally
            if result.messages:
                # Get the last assistant message
                for message in result.messages:
                    print(f"[{message.role}] ", end="")
                    print(" | ".join(
                        content.text for content in message.contents if hasattr(content, 'text') and content.text
                    ))
                try:
                    print(result.output_text)
                except:
                    print("(cant print result.output_text)")
                last_message = result.messages[-1]
                
                # Extract text content
                response_text = ""
                if hasattr(last_message, 'contents'):
                    for content in last_message.contents:
                        if hasattr(content, 'text') and content.text:
                            response_text += content.text
                
                if response_text:
                    print(response_text)
                else:
                    print("(No text response - check tool calls)")
            else:
                print("(No response)")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"   Type: {type(e).__name__}")
        
        print()  # Empty line for readability


async def demo_mode():
    """Run automated demo queries."""
    print("="*70)
    print("🌤️  Weather Agent - Demo Mode")
    print("="*70)
    
    test_queries = [
        "What's the weather like in Seattle?",
        "Give me the temperature in Paris",
        "How's the weather in Tokyo and London?",
        "What's the temperature in New York in Fahrenheit?",
        "Hello! Can you help me with weather info?",
    ]
    
    for query in test_queries:
        print(f"\n{'─'*70}")
        print(f"User: {query}")
        print('─'*70)
        
        try:
            result = await agent.run(query)
            
            # Extract response
            if result.messages:
                last_message = result.messages[-1]
                response_text = ""
                
                if hasattr(last_message, 'contents'):
                    for content in last_message.contents:
                        if hasattr(content, 'text') and content.text:
                            response_text += content.text
                
                print(f"Agent: {response_text if response_text else '(Processing...)'}")
            
            # Show tool calls if any
            print(f"\n📊 Debug Info:")
            print(f"   Messages: {len(result.messages)}")
            for i, msg in enumerate(result.messages):
                print(f"   Message {i}: role={msg.role}, content_count={len(msg.contents)}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
        
        await asyncio.sleep(1)  # Rate limiting


async def main():
    """Main entry point - choose mode."""
    import sys
    
    print("\n🤖 Weather Agent with Tools\n")
    print("Choose mode:")
    print("  1 - Interactive chat")
    print("  2 - Demo mode (automated queries)")
    print("  q - Quit\n")
    
    choice = input("Select (1/2/q): ").strip()
    
    if choice == '1':
        await chat_loop()
    elif choice == '2':
        await demo_mode()
    else:
        print("👋 Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")