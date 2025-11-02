"""
MCP Server implementation for Restaurant Booking System.

This server exposes 5 core tools:
1. search_restaurants - Find restaurants by filters
2. check_availability - Check table availability
3. query_knowledge_base - RAG tool for searching restaurant documents
4. make_reservation - Create bookings
5. manage_booking_state - Read/update booking state
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, ImageContent, EmbeddedResource

# Import tools from external module
from tools import TOOLS, execute_tool

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RestaurantBookingServer:
    """MCP Server for Restaurant Booking System."""

    def __init__(self, server_name: str = "restaurant-booking-server"):
        """
        Initialize the MCP server.

        Args:
            server_name: Name of the server
        """
        self.server_name = server_name
        self.server = Server(server_name)

        self._initialize_server_capabilities()

    def _initialize_server_capabilities(self) -> None:
        """Initialize server tools and handlers."""

        @self.server.list_tools()
        async def handle_list_tools():
            """List available tools."""
            return TOOLS

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict
        ) -> list[TextContent | ImageContent | EmbeddedResource]:
            """
            Handle tool execution requests.

            Args:
                name: Tool name to execute
                arguments: Tool arguments

            Returns:
                List of response content
            """
            try:
                logger.info(f"Executing tool: {name}")
                logger.debug(f"Arguments: {arguments}")

                result = await execute_tool(name, arguments)

                logger.info(f"Tool {name} executed successfully")
                return result

            except ValueError as error:
                logger.error(f"Invalid tool request: {error}")
                return [TextContent(type="text", text=f"Error: {str(error)}")]

            except Exception as error:
                logger.error(f"Tool execution error: {error}", exc_info=True)
                return [TextContent(type="text", text=f"Error executing tool: {str(error)}")]

    def write_mcp_config(self) -> str:
        """
        Write MCP configuration file for client integration.

        Returns:
            Path to the configuration file
        """
        config = {
            "mcpServers": {
                "restaurant-booking-server": {
                    "command": "python",
                    "args": ["-m", "server"],
                    "description": "Restaurant Booking System with search, availability, RAG, and booking management",
                    "capabilities": {
                        "tools": [
                            {
                                "name": tool.name,
                                "description": tool.description
                            }
                            for tool in TOOLS
                        ]
                    },
                    "env": {}
                }
            }
        }

        config_path = Path("mcp.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        logger.info(f"MCP configuration written to {config_path.resolve()}")
        return str(config_path)

    async def run(self) -> None:
        """Run the MCP server."""
        # Write MCP configuration file
        config_path = self.write_mcp_config()

        logger.info("=" * 60)
        logger.info("Restaurant Booking MCP Server")
        logger.info("=" * 60)
        logger.info(f"Server running on stdio")
        logger.info(f"Configuration: {config_path}")
        logger.info(f"Available tools: {len(TOOLS)}")
        for tool in TOOLS:
            logger.info(f"  • {tool.name}: {tool.description}")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop the server")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main() -> None:
    """Main function to start the server."""
    server = RestaurantBookingServer()

    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
    except Exception as error:
        logger.error(f"Server error: {error}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
