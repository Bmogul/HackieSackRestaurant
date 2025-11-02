#!/usr/bin/env python3
"""
AWS Bedrock Agent for Restaurant Booking System.

This agent uses AWS Bedrock's Claude model to interact with MCP tools,
providing a fully cloud-based AI solution.
"""

import asyncio
import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

import boto3
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

# Load environment
load_dotenv()


class MCPToolWrapper:
    """
    Wrapper to connect AWS Bedrock to MCP server.
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

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Get list of available MCP tools."""
        try:
            tools_result = await self.mcp_session.list_tools()

            tools = []
            for tool in tools_result.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })

            return tools
        except Exception as e:
            print(f"Error listing tools: {e}")
            return []


class AWSBedrockAgent:
    """
    AWS Bedrock-powered AI agent for restaurant booking.

    Features:
    - Uses Claude via AWS Bedrock (no OpenAI needed)
    - Tool calling with Bedrock Converse API
    - Fully serverless architecture
    - Pay-per-use pricing
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
        region: str = None
    ):
        """
        Initialize the AWS Bedrock agent.

        Args:
            model_id: Bedrock model ID (Claude recommended)
            region: AWS region (defaults to AWS_REGION env var)
        """
        self.model_id = model_id
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.mcp_wrapper = MCPToolWrapper()
        self.conversation_history = []

        # Initialize Bedrock client
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        print(f"✓ AWS Bedrock client initialized (region: {self.region})")
        print(f"  Model: {model_id}")

    async def initialize(self):
        """Initialize MCP connection and load tools."""
        await self.mcp_wrapper.connect()

        # Get MCP tools and convert to Bedrock tool format
        self.mcp_tools = await self.mcp_wrapper.list_tools()
        self.bedrock_tools = self._convert_tools_to_bedrock_format()

        print(f"✓ Loaded {len(self.mcp_tools)} MCP tools")

    def _convert_tools_to_bedrock_format(self) -> List[Dict[str, Any]]:
        """Convert MCP tool schemas to Bedrock tool format."""
        bedrock_tools = []

        for tool in self.mcp_tools:
            bedrock_tool = {
                "toolSpec": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "inputSchema": {
                        "json": tool["input_schema"]
                    }
                }
            }
            bedrock_tools.append(bedrock_tool)

        return bedrock_tools

    async def _call_bedrock_converse(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """Call Bedrock Converse API with tool support."""

        converse_params = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": 2000,
                "temperature": 0.3,
            },
            "toolConfig": {
                "tools": self.bedrock_tools
            }
        }

        if system_prompt:
            converse_params["system"] = [{"text": system_prompt}]

        try:
            response = await asyncio.to_thread(
                self.bedrock.converse,
                **converse_params
            )
            return response
        except Exception as e:
            print(f"Bedrock API error: {e}")
            raise

    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return agent response.

        Args:
            user_message: User's input text

        Returns:
            Agent's response
        """
        system_prompt = """You are a helpful restaurant booking assistant.

Use the available tools to help users:
- Search for restaurants (by cuisine, location, price, etc.)
- Query the knowledge base for natural language searches (dishes, ambiance, features)
- Check availability at restaurants
- Make reservations
- Manage bookings

Always provide concise, friendly responses. When showing restaurant results, focus on the most relevant 2-3 options."""

        # Build conversation messages
        messages = []

        # Add conversation history (last 5 exchanges)
        for exchange in self.conversation_history[-5:]:
            messages.append({
                "role": "user",
                "content": [{"text": exchange["user"]}]
            })
            messages.append({
                "role": "assistant",
                "content": [{"text": exchange["agent"]}]
            })

        # Add current user message
        messages.append({
            "role": "user",
            "content": [{"text": user_message}]
        })

        try:
            # Initial Bedrock call
            response = await self._call_bedrock_converse(messages, system_prompt)

            # Process response and handle tool calls
            final_response = await self._process_bedrock_response(
                response,
                messages,
                system_prompt
            )

            # Update conversation history
            self.conversation_history.append({
                "user": user_message,
                "agent": final_response,
                "timestamp": datetime.now().isoformat()
            })

            return final_response

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    async def _process_bedrock_response(
        self,
        response: Dict[str, Any],
        messages: List[Dict[str, Any]],
        system_prompt: str,
        max_iterations: int = 5
    ) -> str:
        """
        Process Bedrock response, handling tool calls in a loop.

        This implements the agentic loop:
        1. Bedrock decides to use a tool
        2. We execute the tool via MCP
        3. Send results back to Bedrock
        4. Repeat until Bedrock provides final answer
        """

        iteration = 0
        current_response = response

        while iteration < max_iterations:
            stop_reason = current_response.get("stopReason")

            # If tool use requested
            if stop_reason == "tool_use":
                print(f"[Iteration {iteration + 1}] Bedrock requested tool use")

                # Extract tool use requests from response
                tool_results = []
                tool_use_blocks = []

                for content_block in current_response["output"]["message"]["content"]:
                    if "toolUse" in content_block:
                        tool_use = content_block["toolUse"]
                        tool_use_blocks.append(content_block)

                        tool_name = tool_use["name"]
                        tool_input = tool_use["input"]
                        tool_use_id = tool_use["toolUseId"]

                        print(f"  Tool: {tool_name}")
                        print(f"  Input: {json.dumps(tool_input, indent=2)}")

                        # Execute tool via MCP
                        tool_output = await self.mcp_wrapper.call_tool(tool_name, tool_input)

                        print(f"  Output: {tool_output[:200]}...")

                        # Format tool result for Bedrock
                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [{"text": tool_output}]
                            }
                        })

                # Add assistant's tool use to messages
                messages.append({
                    "role": "assistant",
                    "content": tool_use_blocks
                })

                # Add tool results to messages
                messages.append({
                    "role": "user",
                    "content": tool_results
                })

                # Call Bedrock again with tool results
                current_response = await self._call_bedrock_converse(messages, system_prompt)
                iteration += 1

            # If end turn (final answer)
            elif stop_reason == "end_turn":
                # Extract text response
                text_parts = []
                for content_block in current_response["output"]["message"]["content"]:
                    if "text" in content_block:
                        text_parts.append(content_block["text"])

                return "\n".join(text_parts) if text_parts else "No response generated."

            else:
                # Unexpected stop reason
                print(f"Unexpected stop reason: {stop_reason}")
                return "I encountered an unexpected issue. Please try again."

        return "I reached the maximum number of tool iterations. Please simplify your request."

    async def run_interactive(self):
        """Run interactive conversation loop."""
        print("\n" + "="*70)
        print("🤖 Restaurant Booking AI Agent (AWS Bedrock)")
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

    parser = argparse.ArgumentParser(description="AWS Bedrock Restaurant Booking Agent")
    parser.add_argument(
        "--model",
        default="anthropic.claude-3-haiku-20240307-v1:0",
        help="Bedrock model ID"
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (defaults to AWS_REGION env var)"
    )

    args = parser.parse_args()

    try:
        agent = AWSBedrockAgent(
            model_id=args.model,
            region=args.region
        )

        await agent.run_interactive()

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        print("Make sure you have:")
        print("  1. AWS credentials configured (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        print("  2. AWS_REGION set in .env")
        print("  3. Bedrock model access enabled in your AWS account")
        print("  4. MCP server module accessible: python -m server")


if __name__ == "__main__":
    asyncio.run(main())
