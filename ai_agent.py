#!/usr/bin/env python3
"""
Custom AI Agent for Restaurant Booking System.

This agent connects to the MCP server and uses an LLM (OpenAI/Ollama/Claude)
to have natural conversations about finding and booking restaurants.
"""

import asyncio
import os
import json
import re
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Try to import OpenAI (fallback to simple NLP if not available)
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Using simple pattern matching.")


class RestaurantAIAgent:
    """
    AI Agent that uses MCP tools to help users find and book restaurants.

    Features:
    - Natural language understanding
    - Context-aware conversations
    - Automatic tool selection
    - Booking management
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_provider: str = "openai",
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the AI agent.

        Args:
            use_llm: Whether to use an LLM for understanding (vs simple patterns)
            llm_provider: "openai", "ollama", or "anthropic"
            model: Model name to use
        """
        self.use_llm = use_llm and OPENAI_AVAILABLE
        self.llm_provider = llm_provider
        self.model = model
        self.conversation_history = []
        self.user_context = {
            "current_booking": None,
            "last_search": None,
            "preferences": {}
        }

        # Initialize LLM client
        if self.use_llm:
            if llm_provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.llm_client = AsyncOpenAI(api_key=api_key)
                else:
                    print("Warning: OPENAI_API_KEY not set. Falling back to pattern matching.")
                    self.use_llm = False

        # MCP session (will be initialized when needed)
        self.mcp_session = None

    async def connect_to_mcp_server(self):
        """Connect to the MCP server."""
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

    async def understand_intent(self, user_message: str) -> Dict[str, Any]:
        """
        Understand user intent from natural language.

        Returns:
            {
                "intent": "search" | "check_availability" | "reserve" | "cancel" | "query" | "chat",
                "parameters": {...}
            }
        """
        if self.use_llm:
            return await self._understand_intent_llm(user_message)
        else:
            return self._understand_intent_patterns(user_message)

    async def _understand_intent_llm(self, user_message: str) -> Dict[str, Any]:
        """Use LLM to understand intent."""
        system_prompt = """You are an intent classifier for a restaurant booking system.
Analyze the user's message and output JSON with:
{
  "intent": "search" | "check_availability" | "reserve" | "cancel" | "query" | "chat",
  "parameters": {
    "cuisine": "...",
    "location": "...",
    "date": "YYYY-MM-DD",
    "time": "HH:MM",
    "party_size": number,
    "query": "...",
    ...
  }
}

Intents:
- search: User wants to filter restaurants by criteria
- query: User wants to search with natural language (any food/ambiance/features)
- check_availability: User wants to check if tables are available
- reserve: User wants to make a booking
- cancel: User wants to cancel a booking
- chat: Casual conversation

Extract dates/times relative to today ({today}).
"""

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt.format(today=datetime.now().strftime("%Y-%m-%d"))},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"intent": "chat", "parameters": {}}

        except Exception as e:
            print(f"LLM error: {e}")
            return self._understand_intent_patterns(user_message)

    def _understand_intent_patterns(self, user_message: str) -> Dict[str, Any]:
        """Use pattern matching to understand intent (fallback)."""
        msg_lower = user_message.lower()
        intent_data = {"intent": "chat", "parameters": {}}

        # Check for cancellation
        if any(word in msg_lower for word in ["cancel", "cancel my", "delete"]):
            intent_data["intent"] = "cancel"
            return intent_data

        # Check for reservation
        if any(word in msg_lower for word in ["book", "reserve", "reservation", "table for"]):
            intent_data["intent"] = "reserve"
            # Try to extract party size
            party_match = re.search(r'(\d+)\s*(?:people|person|guest|pax)', msg_lower)
            if party_match:
                intent_data["parameters"]["party_size"] = int(party_match.group(1))
            return intent_data

        # Check for availability
        if any(word in msg_lower for word in ["available", "availability", "check", "open"]):
            intent_data["intent"] = "check_availability"
            return intent_data

        # Extract cuisine types
        cuisines = ["italian", "french", "japanese", "chinese", "indian", "mexican", "american", "steakhouse"]
        found_cuisine = None
        for cuisine in cuisines:
            if cuisine in msg_lower:
                found_cuisine = cuisine.title()
                if cuisine == "steakhouse":
                    found_cuisine = "American Steakhouse"
                break

        # Check if it's a search by cuisine
        if found_cuisine or any(word in msg_lower for word in ["find", "search", "show me", "looking for"]):
            intent_data["intent"] = "search"
            if found_cuisine:
                intent_data["parameters"]["cuisine"] = found_cuisine
            return intent_data

        # Otherwise treat as natural language query
        if len(user_message.split()) > 2:  # More than just a couple words
            intent_data["intent"] = "query"
            intent_data["parameters"]["query"] = user_message

        return intent_data

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
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

    async def handle_search_intent(self, parameters: Dict[str, Any]) -> str:
        """Handle search for restaurants."""
        arguments = {k: v for k, v in parameters.items() if v is not None}
        if "limit" not in arguments:
            arguments["limit"] = 5

        result = await self.execute_tool("search_restaurants", arguments)
        self.user_context["last_search"] = parameters
        return result

    async def handle_query_intent(self, parameters: Dict[str, Any]) -> str:
        """Handle natural language query."""
        query = parameters.get("query", "")
        limit = parameters.get("limit", 3)  # Show fewer, more relevant results

        result = await self.execute_tool("query_knowledge_base", {"query": query, "limit": limit})

        # Add helpful intro based on query
        if "pasta" in query.lower():
            intro = "Found restaurants serving pasta:\n\n"
        elif "steak" in query.lower():
            intro = "Found steakhouses:\n\n"
        elif "pizza" in query.lower():
            intro = "Found restaurants with pizza:\n\n"
        else:
            intro = "Here's what I found:\n\n"

        return intro + result

    async def handle_availability_intent(self, parameters: Dict[str, Any]) -> str:
        """Handle availability check."""
        # Try to get restaurant_id from context if not provided
        restaurant_id = parameters.get("restaurant_id")

        if not restaurant_id:
            return "Which restaurant would you like to check? Please provide a restaurant ID or search first."

        date = parameters.get("date", (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"))
        time = parameters.get("time", "19:00")

        result = await self.execute_tool("check_availability", {
            "restaurant_id": restaurant_id,
            "date": date,
            "time": time
        })
        return result

    async def handle_reserve_intent(self, parameters: Dict[str, Any]) -> str:
        """Handle reservation."""
        required = ["restaurant_id", "date", "time", "party_size", "customer_name", "customer_phone"]

        # Check for missing parameters
        missing = [p for p in required if p not in parameters]

        if missing:
            return f"To make a reservation, I need: {', '.join(missing)}. Please provide these details."

        result = await self.execute_tool("make_reservation", parameters)

        # Extract booking ID from result if successful
        booking_id_match = re.search(r'BK\w+', result)
        if booking_id_match:
            self.user_context["current_booking"] = booking_id_match.group()

        return result

    async def handle_cancel_intent(self, parameters: Dict[str, Any]) -> str:
        """Handle cancellation."""
        booking_id = parameters.get("booking_id") or self.user_context.get("current_booking")

        if not booking_id:
            return "Which booking would you like to cancel? Please provide the confirmation number."

        result = await self.execute_tool("manage_booking_state", {
            "action": "cancel",
            "booking_id": booking_id
        })

        if "cancelled successfully" in result.lower():
            self.user_context["current_booking"] = None

        return result

    async def process_message(self, user_message: str) -> str:
        """
        Process a user message and return a response.

        This is the main entry point for the agent.
        """
        # Understand intent
        intent_data = await self.understand_intent(user_message)
        intent = intent_data["intent"]
        parameters = intent_data.get("parameters", {})

        print(f"[Debug] Intent: {intent}, Parameters: {parameters}")

        # Route to appropriate handler
        if intent == "search":
            response = await self.handle_search_intent(parameters)
        elif intent == "query":
            response = await self.handle_query_intent(parameters)
        elif intent == "check_availability":
            response = await self.handle_availability_intent(parameters)
        elif intent == "reserve":
            response = await self.handle_reserve_intent(parameters)
        elif intent == "cancel":
            response = await self.handle_cancel_intent(parameters)
        else:  # chat
            response = "Hello! I can help you find restaurants, check availability, and make reservations. What would you like to do?"

        # Update conversation history
        self.conversation_history.append({
            "user": user_message,
            "agent": response,
            "intent": intent,
            "timestamp": datetime.now().isoformat()
        })

        return response

    async def run_interactive(self):
        """Run interactive conversation loop."""
        print("\n" + "="*70)
        print("🤖 Restaurant Booking AI Agent")
        print("="*70)
        print("\nI can help you:")
        print("  • Search for restaurants")
        print("  • Check availability")
        print("  • Make reservations")
        print("  • Answer questions about restaurants")
        print("\nType 'quit' or 'exit' to end the conversation.")
        print("="*70 + "\n")

        await self.connect_to_mcp_server()

        try:
            while True:
                user_input = input("You: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("\nAgent: Thank you for using the Restaurant Booking System. Goodbye! 👋\n")
                    break

                if not user_input:
                    continue

                response = await self.process_message(user_input)
                print(f"\nAgent: {response}\n")

        finally:
            await self.disconnect()


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Restaurant Booking AI Agent")
    parser.add_argument("--no-llm", action="store_true", help="Use pattern matching instead of LLM")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM model to use")
    parser.add_argument("--provider", default="openai", choices=["openai", "ollama"], help="LLM provider")

    args = parser.parse_args()

    agent = RestaurantAIAgent(
        use_llm=not args.no_llm,
        llm_provider=args.provider,
        model=args.model
    )

    await agent.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
