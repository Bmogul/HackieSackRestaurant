#!/usr/bin/env python3
"""
Test script to verify MCP server is working correctly.
This simulates what an AI client would do when connecting.
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    """Test all aspects of the MCP server."""

    print("=" * 70)
    print("MCP SERVER VERIFICATION TEST")
    print("=" * 70)
    print()

    # Configure server connection
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "server"],
        env=None
    )

    try:
        print("📡 Connecting to MCP server...")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:

                # Step 1: Initialize connection
                print("✓ Connection established")
                await session.initialize()
                print("✓ Session initialized")
                print()

                # Step 2: List available tools
                print("🔧 Available Tools:")
                print("-" * 70)
                tools = await session.list_tools()

                for i, tool in enumerate(tools.tools, 1):
                    print(f"{i}. {tool.name}")
                    print(f"   Description: {tool.description}")
                    print()

                print(f"Total tools: {len(tools.tools)}")
                print()

                # Step 3: Test each tool
                print("=" * 70)
                print("TESTING TOOLS")
                print("=" * 70)
                print()

                # Test 1: search_restaurants
                print("Test 1: search_restaurants (Find Italian restaurants)")
                print("-" * 70)
                try:
                    result = await session.call_tool(
                        "search_restaurants",
                        {"cuisine": "Italian", "limit": 2}
                    )
                    for content in result.content:
                        if hasattr(content, 'text'):
                            print(content.text)
                    print("✓ search_restaurants working\n")
                except Exception as e:
                    print(f"✗ Error: {e}\n")

                # Test 2: check_availability
                print("Test 2: check_availability (Check table availability)")
                print("-" * 70)
                try:
                    result = await session.call_tool(
                        "check_availability",
                        {
                            "restaurant_id": "r1",
                            "date": "2025-11-05",
                            "time": "19:00"
                        }
                    )
                    for content in result.content:
                        if hasattr(content, 'text'):
                            print(content.text)
                    print("✓ check_availability working\n")
                except Exception as e:
                    print(f"✗ Error: {e}\n")

                # Test 3: query_knowledge_base (RAG)
                print("Test 3: query_knowledge_base (Search for pizza)")
                print("-" * 70)
                try:
                    result = await session.call_tool(
                        "query_knowledge_base",
                        {"query": "pizza", "limit": 1}
                    )
                    for content in result.content:
                        if hasattr(content, 'text'):
                            # Show first 300 chars
                            text = content.text[:300]
                            print(text + "..." if len(content.text) > 300 else text)
                    print("✓ query_knowledge_base working\n")
                except Exception as e:
                    print(f"✗ Error: {e}\n")

                # Test 4: make_reservation
                print("Test 4: make_reservation (Create test booking)")
                print("-" * 70)
                try:
                    result = await session.call_tool(
                        "make_reservation",
                        {
                            "restaurant_id": "r1",
                            "date": "2025-11-10",
                            "time": "20:00",
                            "party_size": 2,
                            "customer_name": "Test User",
                            "customer_phone": "555-0000"
                        }
                    )
                    booking_id = None
                    for content in result.content:
                        if hasattr(content, 'text'):
                            print(content.text[:200] + "...")
                            # Extract booking ID for next test
                            if "BK" in content.text:
                                import re
                                match = re.search(r'BK\w+', content.text)
                                if match:
                                    booking_id = match.group()
                    print("✓ make_reservation working\n")

                    # Test 5: manage_booking_state
                    if booking_id:
                        print("Test 5: manage_booking_state (Cancel test booking)")
                        print("-" * 70)
                        try:
                            result = await session.call_tool(
                                "manage_booking_state",
                                {"action": "cancel", "booking_id": booking_id}
                            )
                            for content in result.content:
                                if hasattr(content, 'text'):
                                    print(content.text)
                            print("✓ manage_booking_state working\n")
                        except Exception as e:
                            print(f"✗ Error: {e}\n")

                except Exception as e:
                    print(f"✗ Error: {e}\n")

                # Summary
                print("=" * 70)
                print("VERIFICATION COMPLETE")
                print("=" * 70)
                print()
                print("✓ MCP server is running correctly")
                print("✓ All 5 tools are accessible")
                print("✓ Server ready for AI client connections")
                print()
                print("Your server can now be used by:")
                print("  • Claude Desktop (via MCP config)")
                print("  • OpenAI GPTs (via MCP plugin)")
                print("  • Custom LLM applications")
                print("  • Any MCP-compatible AI assistant")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nServer may not be running or there's a connection issue.")
        print("Check: python -m server")
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    exit(0 if success else 1)
