#!/usr/bin/env python3
"""
LangChain AI Agent for Restaurant Booking System.

This agent uses LangChain to interact with MCP tools,
providing superior natural language understanding and reasoning.
"""

import asyncio
import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Try to import LLM providers
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: langchain-openai not installed. Install with: pip install langchain-openai")

# Load environment
load_dotenv()


class MCPToolWrapper:
    """
    Wrapper to connect LangChain tools to MCP server.
    Manages MCP session lifecycle and tool execution.
    """

    def __init__(self):
        self.mcp_session = None
        self.stdio_context = None
        self.session_context = None

    async def connect(self):
        """Connect to MCP server."""
        server_params = StdioServerParameters(
            command="python",
            args=["-m", "server"],
            env=None
        )

        self.stdio_context = stdio_client(server_params)
        read, write = await self.stdio_context.__aenter__()

        self.session_context = ClientSession(read, write)
        self.mcp_session = await self.session_context.__aenter__()
        await self.mcp_session.initialize()

        print("✓ Connected to MCP Restaurant Booking Server")

    async def disconnect(self):
        """Disconnect from MCP server."""
        if self.session_context:
            await self.session_context.__aexit__(None, None, None)
        if self.stdio_context:
            await self.stdio_context.__aexit__(None, None, None)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute an MCP tool and return the result."""
        try:
            result = await self.mcp_session.call_tool(tool_name, arguments)

            # Extract text from result
            text_parts = []
            for content in result.content:
                if hasattr(content, 'text'):
                    text_parts.append(content.text)

            return "\n".join(text_parts) if text_parts else "No results"

        except Exception as e:
            return f"Error: {str(e)}"


class RestaurantLangChainAgent:
    """
    LangChain-powered AI agent for restaurant booking.

    Features:
    - Smart tool selection
    - Conversation memory
    - Better natural language understanding
    - Concise, user-friendly responses
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.3
    ):
        """
        Initialize the LangChain agent.

        Args:
            model: OpenAI model name
            temperature: LLM temperature (0-1)
        """
        self.model = model
        self.temperature = temperature
        self.mcp_wrapper = MCPToolWrapper()
        self.conversation_history = []

        # Check OpenAI availability
        if not OPENAI_AVAILABLE:
            raise ImportError("langchain-openai required. Install: pip install langchain-openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key
        )

        # Create tool descriptions for the LLM
        self.tools_description = """
Available Tools:
1. query_knowledge_base - Search restaurants using natural language (RAG). Use for queries about specific dishes, ambiance, or features.
   Example: "pasta restaurant", "romantic place with wine", "best steak"

2. search_restaurants - Filter restaurants by structured criteria like cuisine type, location, price range.
   Example: cuisine="Italian", location="Downtown", price_range="$$"

3. check_availability - Check if a restaurant has available tables at a specific date/time.
   Requires: restaurant_id, date (YYYY-MM-DD), time (HH:MM)

4. make_reservation - Create a booking at a restaurant.
   Requires: restaurant_id, date, time, party_size, customer_name, customer_phone

5. manage_booking_state - Get booking details or cancel a reservation.
   Requires: action ("get" or "cancel"), booking_id
"""

    async def initialize(self):
        """Initialize MCP connection."""
        await self.mcp_wrapper.connect()
        print("✓ LangChain agent initialized")

    async def _select_tool_and_params(self, user_message: str) -> Dict[str, Any]:
        """Use LLM to select the appropriate tool and parameters."""

        system_prompt = f"""You are a restaurant booking assistant. Given a user's request, determine which tool to use and what parameters to pass.

{self.tools_description}

Analyze the user's message and respond with JSON:
{{
  "tool": "tool_name",
  "parameters": {{...}},
  "reasoning": "why this tool"
}}

GUIDELINES:
- For queries about specific food/dishes/ambiance: use query_knowledge_base
- For filtering by cuisine/location/price: use search_restaurants
- Always use query_knowledge_base for natural language food queries
- Keep limit to 3 for better focused results
- If user says "want location serving pasta", use query_knowledge_base with query="pasta restaurant"

Current date: {datetime.now().strftime("%Y-%m-%d")}
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                messages
            )

            # Extract JSON from response
            content = response.content
            json_match = content.strip()
            if json_match.startswith('```json'):
                json_match = json_match.split('```json')[1].split('```')[0].strip()
            elif json_match.startswith('```'):
                json_match = json_match.split('```')[1].split('```')[0].strip()

            result = json.loads(json_match)
            return result

        except Exception as e:
            print(f"LLM tool selection error: {e}")
            # Fallback to simple pattern matching
            return self._fallback_tool_selection(user_message)

    def _fallback_tool_selection(self, user_message: str) -> Dict[str, Any]:
        """Fallback tool selection using pattern matching."""
        msg_lower = user_message.lower()

        # Check for specific dishes or natural language queries
        food_keywords = ["pasta", "pizza", "steak", "sushi", "romantic", "wine", "outdoor"]
        if any(keyword in msg_lower for keyword in food_keywords):
            return {
                "tool": "query_knowledge_base",
                "parameters": {"query": user_message, "limit": 3},
                "reasoning": "Natural language food query"
            }

        # Check for cuisine-based search
        cuisines = ["italian", "french", "japanese", "chinese", "indian", "mexican"]
        if any(cuisine in msg_lower for cuisine in cuisines):
            for cuisine in cuisines:
                if cuisine in msg_lower:
                    return {
                        "tool": "search_restaurants",
                        "parameters": {"cuisine": cuisine.title(), "limit": 3},
                        "reasoning": "Cuisine filter search"
                    }

        # Default to knowledge base for general queries
        return {
            "tool": "query_knowledge_base",
            "parameters": {"query": user_message, "limit": 3},
            "reasoning": "General query"
        }

    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Execute the selected tool."""
        return await self.mcp_wrapper.call_tool(tool_name, parameters)

    async def _format_response(self, user_message: str, tool_result: str, tool_used: str) -> str:
        """Use LLM to format a natural, concise response."""

        prompt = f"""You are a friendly restaurant assistant. Format this tool result into a natural, concise response.

User asked: {user_message}
Tool used: {tool_used}
Tool result:
{tool_result}

Provide a friendly, concise response that:
1. Directly answers their question
2. Shows ONLY the top 2-3 most relevant restaurants
3. Highlights why each restaurant matches their request
4. Keeps it brief and scannable

DO NOT include all restaurants if there are many - pick the best matches."""

        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                [HumanMessage(content=prompt)]
            )
            return response.content

        except Exception as e:
            print(f"LLM formatting error: {e}")
            return tool_result  # Return raw result as fallback

    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return agent response.

        Args:
            user_message: User's input text

        Returns:
            Agent's response
        """
        try:
            # Step 1: Select tool and parameters
            tool_selection = await self._select_tool_and_params(user_message)
            print(f"[Debug] Selected tool: {tool_selection['tool']}")
            print(f"[Debug] Parameters: {tool_selection['parameters']}")

            # Step 2: Execute tool
            tool_result = await self._execute_tool(
                tool_selection['tool'],
                tool_selection['parameters']
            )

            # Step 3: Format response
            response = await self._format_response(
                user_message,
                tool_result,
                tool_selection['tool']
            )

            # Update conversation history
            self.conversation_history.append({
                "user": user_message,
                "agent": response,
                "tool": tool_selection['tool'],
                "timestamp": datetime.now().isoformat()
            })

            return response

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    async def run_interactive(self):
        """Run interactive conversation loop."""
        print("\n" + "="*70)
        print("🤖 Restaurant Booking AI Agent (LangChain)")
        print("="*70)
        print("\nI can help you:")
        print("  • Search for restaurants")
        print("  • Check availability")
        print("  • Make reservations")
        print("  • Answer questions about restaurants")
        print("\nType 'quit' or 'exit' to end the conversation.")
        print("="*70 + "\n")

        await self.initialize()

        try:
            while True:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("\nAgent: Thank you for using the Restaurant Booking System. Goodbye! 👋\n")
                    break

                if not user_input:
                    continue

                response = await self.chat(user_input)
                print(f"\nAgent: {response}")

        finally:
            await self.mcp_wrapper.disconnect()


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LangChain Restaurant Booking Agent")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.3, help="LLM temperature (0-1)")

    args = parser.parse_args()

    try:
        agent = RestaurantLangChainAgent(
            model=args.model,
            temperature=args.temperature
        )

        await agent.run_interactive()

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        print("Make sure you have:")
        print("  1. OPENAI_API_KEY set in .env")
        print("  2. langchain-openai installed: pip install langchain-openai")
        print("  3. MCP server module accessible: python -m server")


if __name__ == "__main__":
    asyncio.run(main())
