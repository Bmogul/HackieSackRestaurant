"""MCP Server implementation for data generation."""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    Resource,
    Prompt,
    TextContent,
    ImageContent,
    EmbeddedResource
)

from .Tools.server_tools import (
    create_db,
    create_synthetic_data,
    create_vector_db,
    insert_doc_data_in_db,
    CreateDBInputs,
    CreateSyntheticDataInputs,
    CreateVectorDBInputs,
    InsertDocDataInDBInputs
)
from .Resources.server_resources import get_local_db_schema
from .Prompts.server_prompts import (
    get_generate_sample_data_prompt,
    GenerateSampleDataPromptInputs
)
from .logger import loggers

# Import local restaurant/availability helpers
from Data import helpers as data_helpers

logger = loggers.server()


class McpDataGenServer:
    """MCP Data Generation Server."""
    
    def __init__(self, server_name: str = "mcp-datagen-server"):
        """
        Initialize the MCP server.
        
        Args:
            server_name: Name of the server
        """
        self.server_name = server_name
        self.server = Server(server_name)
        self.vector_db: Optional[Any] = None
        self.document_names: list = []
        
        self._initialize_server_capabilities()
    
    def _initialize_server_capabilities(self) -> None:
        """Initialize server tools, resources, and prompts."""
        
        # Register tools
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools."""
            # Expose restaurant and booking related tools
            return [
                Tool(
                    name="search_restaurants",
                    description="Search restaurants by filters (cuisine, location, price_range, dietary_options, min_rating)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "cuisine": {"type": "string"},
                            "location": {"type": "string"},
                            "price_range": {"type": "string"},
                            "dietary_options": {"type": "array", "items": {"type": "string"}},
                            "min_rating": {"type": "number"},
                            "limit": {"type": "integer"}
                        }
                    }
                ),
                Tool(
                    name="check_availabilty",
                    description="Check availability for a restaurant at a specific date and time",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "restaurant_id": {"type": "string"},
                            "date": {"type": "string", "description": "YYYY-MM-DD"},
                            "time": {"type": "string", "description": "HH:MM"}
                        },
                        "required": ["restaurant_id", "date", "time"]
                    }
                ),
                Tool(
                    name="query_knowledge_base",
                    description="Simple knowledge-base query over restaurant data (name, features, dishes)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="make_reservation",
                    description="Create a reservation (restaurant_id, date, time, party_size, customer_name, customer_phone)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "restaurant_id": {"type": "string"},
                            "date": {"type": "string"},
                            "time": {"type": "string"},
                            "party_size": {"type": "integer"},
                            "customer_name": {"type": "string"},
                            "customer_phone": {"type": "string"},
                            "special_requests": {"type": "string"}
                        },
                        "required": ["restaurant_id", "date", "time", "party_size", "customer_name", "customer_phone"]
                    }
                ),
                Tool(
                    name="manage_booking_state",
                    description="Manage booking state: get, cancel (provide action and booking_id)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["get", "cancel"]},
                            "booking_id": {"type": "string"}
                        },
                        "required": ["action"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
            """Handle tool calls."""
            try:
                # Map tool names to functionality implemented in Data/helpers.py
                # Note: helpers functions are synchronous; call them directly (they are lightweight file ops)
                if name == "search_restaurants":
                    restaurants = data_helpers.load_restaurants(os.path.join('Data','restaurants.json'))
                    cuisine = arguments.get('cuisine')
                    location = arguments.get('location')
                    price_range = arguments.get('price_range')
                    dietary_options = arguments.get('dietary_options')
                    min_rating = arguments.get('min_rating')
                    limit = arguments.get('limit', 10)

                    results = data_helpers.search_restaurants(
                        restaurants,
                        cuisine=cuisine,
                        location=location,
                        price_range=price_range,
                        dietary_options=dietary_options,
                        min_rating=min_rating
                    )

                    # Format output
                    formatted = []
                    for r in results[:limit]:
                        formatted.append(f"{r['name']} ({r['id']}) — {r['cuisine']} — {r['location']} — Rating: {r['rating']}")

                    text = "\n".join(formatted) if formatted else "No restaurants found matching filters."
                    return [TextContent(type="text", text=text)]

                elif name == "check_availabilty":
                    availability = data_helpers.load_availability(os.path.join('Data','availability.json'))
                    restaurant_id = arguments.get('restaurant_id')
                    date = arguments.get('date')
                    time = arguments.get('time')

                    tables = data_helpers.check_availability(availability, restaurant_id, date, time)
                    if tables is None:
                        text = f"No availability data for restaurant {restaurant_id} at {date} {time}"
                    else:
                        text = f"Restaurant {restaurant_id} has {tables} table(s) available at {date} {time}"
                    return [TextContent(type="text", text=text)]

                elif name == "query_knowledge_base":
                    query = (arguments.get('query') or '').lower()
                    limit = arguments.get('limit', 5)
                    restaurants = data_helpers.load_restaurants(os.path.join('Data','restaurants.json'))

                    matches = []
                    for r in restaurants:
                        hay = " ".join([
                            r.get('name',''), r.get('cuisine',''), r.get('features',''), ' '.join(r.get('popular_dishes',[]))
                        ]).lower()
                        if query in hay:
                            matches.append(r)

                    if not matches:
                        text = "No knowledge-base matches found."
                    else:
                        text = "\n".join([data_helpers.format_restaurant_display(m) for m in matches[:limit]])

                    return [TextContent(type="text", text=text)]

                elif name == "make_reservation":
                    # Required args
                    restaurant_id = arguments.get('restaurant_id')
                    date = arguments.get('date')
                    time = arguments.get('time')
                    party_size = int(arguments.get('party_size')) if arguments.get('party_size') is not None else None
                    customer_name = arguments.get('customer_name')
                    customer_phone = arguments.get('customer_phone')
                    special_requests = arguments.get('special_requests')

                    # Validate
                    ok, errors = data_helpers.validate_booking(date, time, party_size)
                    if not ok:
                        return [TextContent(type="text", text=f"Booking validation failed: {errors}")]

                    restaurants = data_helpers.load_restaurants(os.path.join('Data','restaurants.json'))
                    restaurant = data_helpers.get_restaurant_by_id(restaurants, restaurant_id)
                    if restaurant is None:
                        return [TextContent(type="text", text=f"Restaurant {restaurant_id} not found")]

                    # Check availability
                    availability = data_helpers.load_availability(os.path.join('Data','availability.json'))
                    tables = data_helpers.check_availability(availability, restaurant_id, date, time) or 0
                    if tables < 1:
                        # Try to find alternatives
                        alternatives = data_helpers.find_alternative_slots(availability, restaurant_id, date, time)
                        alt_text = "" if not alternatives else "Alternative slots:\n" + "\n".join([f"{a['time']} ({a['available_tables']})" for a in alternatives])
                        return [TextContent(type="text", text=f"No tables available at requested time. {alt_text}")]

                    # Create booking
                    booking = data_helpers.create_booking(
                        restaurant_id=restaurant_id,
                        restaurant_name=restaurant['name'],
                        date=date,
                        time=time,
                        party_size=party_size,
                        customer_name=customer_name,
                        customer_phone=customer_phone,
                        special_requests=special_requests
                    )

                    # Update availability (book 1 table)
                    updated = data_helpers.update_availability_after_booking(availability, restaurant_id, date, time, tables_booked=1)
                    avail_path = os.path.join('Data','availability.json')
                    data_helpers.save_to_file(updated, avail_path)

                    # Append booking to bookings file
                    bookings_path = os.path.join('Data','bookings.json')
                    try:
                        with open(bookings_path, 'r', encoding='utf-8') as bf:
                            bookings = json.load(bf)
                    except Exception:
                        bookings = []

                    bookings.append(booking)
                    data_helpers.save_to_file(bookings, bookings_path)

                    text = data_helpers.format_booking_confirmation(booking)
                    return [TextContent(type="text", text=text)]

                elif name == "manage_booking_state":
                    action = arguments.get('action')
                    booking_id = arguments.get('booking_id')
                    bookings_path = os.path.join('Data','bookings.json')
                    try:
                        with open(bookings_path, 'r', encoding='utf-8') as bf:
                            bookings = json.load(bf)
                    except Exception:
                        bookings = []

                    if action == 'get':
                        if booking_id:
                            found = next((b for b in bookings if b['booking_id'] == booking_id), None)
                            if not found:
                                return [TextContent(type="text", text=f"Booking {booking_id} not found")]
                            return [TextContent(type="text", text=data_helpers.format_booking_confirmation(found))]
                        else:
                            return [TextContent(type="text", text=json.dumps(bookings, indent=2))]

                    elif action == 'cancel':
                        if not booking_id:
                            return [TextContent(type="text", text="booking_id required to cancel")]
                        idx = next((i for i,b in enumerate(bookings) if b['booking_id']==booking_id), None)
                        if idx is None:
                            return [TextContent(type="text", text=f"Booking {booking_id} not found")]

                        # Mark cancelled
                        bookings[idx]['status'] = 'cancelled'

                        # Return a table to availability (naive +1)
                        try:
                            availability = data_helpers.load_availability(os.path.join('Data','availability.json'))
                            r_id = bookings[idx]['restaurant_id']
                            d = bookings[idx]['date']
                            t = bookings[idx]['time']
                            availability.setdefault(r_id, {}).setdefault(d, {})[t] = availability.get(r_id, {}).get(d, {}).get(t, 0) + 1
                            data_helpers.save_to_file(availability, os.path.join('Data','availability.json'))
                        except Exception:
                            pass

                        data_helpers.save_to_file(bookings, bookings_path)
                        return [TextContent(type="text", text=f"Booking {booking_id} cancelled")]

                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                # Fallback - should not reach here
                return [TextContent(type="text", text="Tool executed")]
                
            except Exception as error:
                logger.error(f"Tool execution error: {error}")
                return [TextContent(type="text", text=f"Error: {str(error)}")]
        
        # Register resources
        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """List available resources."""
            return [
                Resource(
                    uri="sqlite://schema",
                    name="DbSchema",
                    description="DB Schema Text",
                    mimeType="text/plain"
                )
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri) -> str:
            """Handle resource reading."""
            try:
                # Convert URI to string if it's not already
                uri_str = str(uri)

                
                if uri_str == "sqlite://schema":
                    result = await get_local_db_schema(uri_str)
                    return result["contents"][0]["text"]
                else:
                    raise ValueError(f"Unknown resource: {uri_str}")
                    
            except Exception as error:
                logger.error(f"Resource reading error: {error}")
                return f"Error reading resource: {str(error)}"
        
        # Register prompts
        @self.server.list_prompts()
        async def handle_list_prompts() -> list[Prompt]:
            """List available prompts."""
            return [
                Prompt(
                    name="generate-sample-data-csv",
                    description="Generate synthetic data based on schema and rules",
                    arguments=[
                        {
                            "name": "dbSchema",
                            "description": "The database schema definition",
                            "required": True
                        },
                        {
                            "name": "recordCount",
                            "description": "The number of records to generate",
                            "required": True
                        }
                    ]
                )
            ]
        
        @self.server.get_prompt()
        async def handle_get_prompt(name: str, arguments: dict) -> dict:
            """Handle prompt requests."""
            try:
                if name == "generate-sample-data-csv":
                    inputs = GenerateSampleDataPromptInputs(**arguments)
                    return await get_generate_sample_data_prompt(inputs)
                else:
                    raise ValueError(f"Unknown prompt: {name}")
                    
            except Exception as error:
                logger.error(f"Prompt generation error: {error}")
                return {
                    "description": "Error generating prompt",
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": f"Error: {str(error)}"
                            }
                        }
                    ]
                }
    
    def write_mcp_config(self) -> str:
        """
        Write MCP configuration file.
        
        Returns:
            Path to the configuration file
        """
        config = {
            "mcpServers": {
                "mcp-datagen-server": {
                    "command": "uv",
                    "args": ["run", "python", "-m", "src.server"],
                    "description": "MCP Database Server with SQLite and Vector DB support",
                    "capabilities": {
                        "tools": [
                            {"name": "search_restaurants", "description": "Search restaurants by filters"},
                            {"name": "check_availabilty", "description": "Check availability for a restaurant at a specific date/time"},
                            {"name": "query_knowledge_base", "description": "Query restaurant knowledge base"},
                            {"name": "make_reservation", "description": "Create a reservation"},
                            {"name": "manage_booking_state", "description": "Get or cancel bookings"}
                        ],
                        "resources": [
                            {
                                "uri": "sqlite://schema",
                                "name": "DBSchema"
                            }
                        ],
                        "prompts": [
                            {
                                "name": "generate-sample-data-csv",
                                "description": "Generate synthetic data based on schema and rules"
                            }
                        ]
                    },
                    "env": {}
                }
            }
        }
        
        config_path = Path("mcp.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"📝 MCP configuration written to {config_path.resolve()}")
        return str(config_path)
    
    async def run(self) -> None:
        """Run the server."""
        # Write MCP configuration file
        config_path = self.write_mcp_config()
        
        logger.info("📡 Server is running on stdio - connect with an MCP client")
        logger.info(f"🔧 Server configuration available in: {config_path}")
        logger.info("🛑 Press Ctrl+C to stop the server")
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main() -> None:
    """Main function to start the server."""
    logger.info("🚀 Starting MCP Data Generation Server...")
    
    server = McpDataGenServer()
    
    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
    except Exception as error:
        logger.error(f"❌ Server error: {error}")
        raise


if __name__ == "__main__":
    asyncio.run(main())