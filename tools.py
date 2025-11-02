"""
MCP Tools for Restaurant Booking Server.

This module defines the 5 core tools and their implementation handlers.
"""

import os
import json
from typing import Dict, Any, List
from mcp.types import Tool, TextContent

from Data import helpers as data_helpers


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

TOOLS = [
    Tool(
        name="search_restaurants",
        description="Search and filter restaurants by cuisine, location, price range, dietary options, and rating",
        inputSchema={
            "type": "object",
            "properties": {
                "cuisine": {
                    "type": "string",
                    "description": "Filter by cuisine type (e.g., Italian, Japanese, Indian)"
                },
                "location": {
                    "type": "string",
                    "description": "Filter by location/neighborhood"
                },
                "price_range": {
                    "type": "string",
                    "description": "Filter by price range ($, $$, $$$, $$$$)"
                },
                "dietary_options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by dietary options (vegetarian, vegan, gluten-free, etc.)"
                },
                "min_rating": {
                    "type": "number",
                    "description": "Minimum rating (0-5)",
                    "minimum": 0,
                    "maximum": 5
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10
                }
            }
        }
    ),
    Tool(
        name="check_availability",
        description="Check table availability for a specific restaurant at a given date and time",
        inputSchema={
            "type": "object",
            "properties": {
                "restaurant_id": {
                    "type": "string",
                    "description": "Unique restaurant identifier"
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format",
                    "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
                },
                "time": {
                    "type": "string",
                    "description": "Time in HH:MM format (24-hour)",
                    "pattern": "^\\d{2}:\\d{2}$"
                }
            },
            "required": ["restaurant_id", "date", "time"]
        }
    ),
    Tool(
        name="query_knowledge_base",
        description="RAG tool - Search restaurant documents and knowledge base using semantic search. Query restaurant information, menus, reviews, and features.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="make_reservation",
        description="Create a new restaurant reservation with customer details",
        inputSchema={
            "type": "object",
            "properties": {
                "restaurant_id": {
                    "type": "string",
                    "description": "Restaurant identifier"
                },
                "date": {
                    "type": "string",
                    "description": "Reservation date (YYYY-MM-DD)"
                },
                "time": {
                    "type": "string",
                    "description": "Reservation time (HH:MM)"
                },
                "party_size": {
                    "type": "integer",
                    "description": "Number of guests (1-20)",
                    "minimum": 1,
                    "maximum": 20
                },
                "customer_name": {
                    "type": "string",
                    "description": "Customer full name"
                },
                "customer_phone": {
                    "type": "string",
                    "description": "Customer phone number"
                },
                "special_requests": {
                    "type": "string",
                    "description": "Any special requests or dietary requirements"
                }
            },
            "required": ["restaurant_id", "date", "time", "party_size", "customer_name", "customer_phone"]
        }
    ),
    Tool(
        name="manage_booking_state",
        description="Read or update booking state - retrieve booking details or cancel reservations",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "cancel"],
                    "description": "Action to perform: 'get' to retrieve booking(s), 'cancel' to cancel a booking"
                },
                "booking_id": {
                    "type": "string",
                    "description": "Booking confirmation number (required for specific operations)"
                }
            },
            "required": ["action"]
        }
    )
]


# ============================================================================
# TOOL HANDLERS
# ============================================================================

async def handle_search_restaurants(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Handler for search_restaurants tool.

    Args:
        arguments: Tool arguments containing search filters

    Returns:
        List of TextContent with search results
    """
    restaurants = data_helpers.load_restaurants(os.path.join('Data', 'restaurants.json'))

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
    if not results:
        text = "No restaurants found matching filters."
    else:
        formatted = []
        for r in results[:limit]:
            formatted.append(
                f"{r['name']} ({r['id']}) — {r['cuisine']} — {r['location']} — "
                f"Rating: {r['rating']} — {r['price_range']}"
            )
        text = "\n".join(formatted)

    return [TextContent(type="text", text=text)]


async def handle_check_availability(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Handler for check_availability tool.

    Args:
        arguments: Tool arguments with restaurant_id, date, time

    Returns:
        List of TextContent with availability information
    """
    availability = data_helpers.load_availability(os.path.join('Data', 'availability.json'))

    restaurant_id = arguments.get('restaurant_id')
    date = arguments.get('date')
    time = arguments.get('time')

    tables = data_helpers.check_availability(availability, restaurant_id, date, time)

    if tables is None:
        text = f"No availability data for restaurant {restaurant_id} at {date} {time}"
    else:
        if tables > 0:
            text = f"✓ Restaurant {restaurant_id} has {tables} table(s) available at {date} {time}"
        else:
            text = f"✗ No tables available at {date} {time}"
            # Suggest alternatives
            alternatives = data_helpers.find_alternative_slots(availability, restaurant_id, date, time, max_alternatives=3)
            if alternatives:
                text += "\n\nAlternative time slots:"
                for alt in alternatives:
                    text += f"\n  • {alt['time']} - {alt['available_tables']} table(s) available"

    return [TextContent(type="text", text=text)]


async def handle_query_knowledge_base(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Handler for query_knowledge_base tool (RAG).

    Uses ChromaDB for semantic search over restaurant documents.

    Args:
        arguments: Tool arguments with query string

    Returns:
        List of TextContent with search results
    """
    query = arguments.get('query', '')
    limit = arguments.get('limit', 5)

    if not query:
        return [TextContent(type="text", text="Please provide a search query.")]

    try:
        # Import RAG module
        from rag import search_knowledge_base

        # Perform hybrid semantic search (local + AWS)
        results = search_knowledge_base(query, limit=limit, use_aws=True)

        if not results:
            text = "No restaurants found matching your query."
        else:
            # Load full restaurant data for detailed display
            restaurants = data_helpers.load_restaurants(os.path.join('Data', 'restaurants.json'))
            restaurant_map = {r['id']: r for r in restaurants}

            result_lines = ["🔍 Hybrid Search Results (Local + AWS Bedrock):\n"]
            for i, result in enumerate(results, 1):
                # Get full restaurant data if available locally
                restaurant = restaurant_map.get(result['id'])

                if restaurant:
                    # Full structured data available
                    result_lines.append(f"{i}. {data_helpers.format_restaurant_display(restaurant)}")
                else:
                    # AWS-only result, display what we have
                    result_lines.append(f"{i}. {result.get('name', 'Unknown')} ({result['id']})")
                    if result.get('cuisine'):
                        result_lines.append(f"   Cuisine: {result['cuisine']}")
                    if result.get('location'):
                        result_lines.append(f"   Location: {result['location']}")

                # Add relevance and source info
                if result.get('relevance_score') is not None:
                    result_lines.append(f"   Relevance: {result['relevance_score']:.1%}")

                source = result.get('source', 'unknown')
                source_emoji = {
                    'local': '💾',
                    'aws_bedrock': '☁️',
                    'hybrid': '🔄'
                }.get(source, '❓')
                result_lines.append(f"   Source: {source_emoji} {source}")

                # Include AWS enrichment if available
                if result.get('aws_content'):
                    aws_snippet = result['aws_content'][:200].replace('\n', ' ')
                    result_lines.append(f"   AWS Context: {aws_snippet}...")

                result_lines.append("-" * 60)

            text = "\n".join(result_lines)

        return [TextContent(type="text", text=text)]

    except Exception as e:
        # Fallback to simple keyword search if RAG fails
        import logging
        logging.warning(f"RAG search failed, falling back to keyword search: {e}")

        query_lower = query.lower()
        restaurants = data_helpers.load_restaurants(os.path.join('Data', 'restaurants.json'))

        matches = []
        for r in restaurants:
            # Build searchable text from restaurant data
            features = ' '.join(r.get('features', [])) if isinstance(r.get('features'), list) else r.get('features', '')
            popular_dishes = ' '.join(r.get('popular_dishes', []))

            searchable_text = " ".join([
                r.get('name', ''),
                r.get('cuisine', ''),
                features,
                popular_dishes,
                r.get('address', ''),
                r.get('location', '')
            ]).lower()

            if query_lower in searchable_text:
                matches.append(r)

        if not matches:
            text = "No restaurants found matching your query."
        else:
            result_lines = ["🔍 Search Results (keyword search):\n"]
            for m in matches[:limit]:
                result_lines.append(data_helpers.format_restaurant_display(m))
                result_lines.append("-" * 60)
            text = "\n".join(result_lines)

        return [TextContent(type="text", text=text)]


async def handle_make_reservation(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Handler for make_reservation tool.

    Args:
        arguments: Tool arguments with reservation details

    Returns:
        List of TextContent with confirmation or error
    """
    # Extract arguments
    restaurant_id = arguments.get('restaurant_id')
    date = arguments.get('date')
    time = arguments.get('time')
    party_size = int(arguments.get('party_size')) if arguments.get('party_size') is not None else None
    customer_name = arguments.get('customer_name')
    customer_phone = arguments.get('customer_phone')
    special_requests = arguments.get('special_requests')

    # Validate booking
    ok, errors = data_helpers.validate_booking(date, time, party_size)
    if not ok:
        return [TextContent(type="text", text=f"❌ Booking validation failed:\n" + "\n".join(f"  • {err}" for err in errors))]

    # Check restaurant exists
    restaurants = data_helpers.load_restaurants(os.path.join('Data', 'restaurants.json'))
    restaurant = data_helpers.get_restaurant_by_id(restaurants, restaurant_id)
    if restaurant is None:
        return [TextContent(type="text", text=f"❌ Restaurant {restaurant_id} not found")]

    # Check availability
    availability = data_helpers.load_availability(os.path.join('Data', 'availability.json'))
    tables = data_helpers.check_availability(availability, restaurant_id, date, time) or 0

    if tables < 1:
        # Try to find alternatives
        alternatives = data_helpers.find_alternative_slots(availability, restaurant_id, date, time, max_alternatives=5)
        if not alternatives:
            text = f"❌ No tables available at {time} on {date} and no alternative slots found."
        else:
            text = f"❌ No tables available at {time} on {date}.\n\nAlternative slots at {restaurant['name']}:"
            for alt in alternatives:
                text += f"\n  • {alt['time']} - {alt['available_tables']} table(s) available"
        return [TextContent(type="text", text=text)]

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

    # Update availability
    updated = data_helpers.update_availability_after_booking(
        availability, restaurant_id, date, time, tables_booked=1
    )
    data_helpers.save_to_file(updated, os.path.join('Data', 'availability.json'))

    # Save booking
    bookings_path = os.path.join('Data', 'bookings.json')
    try:
        with open(bookings_path, 'r', encoding='utf-8') as bf:
            bookings = json.load(bf)
    except Exception:
        bookings = []

    bookings.append(booking)
    data_helpers.save_to_file(bookings, bookings_path)

    text = "✓ " + data_helpers.format_booking_confirmation(booking)
    return [TextContent(type="text", text=text)]


async def handle_manage_booking_state(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Handler for manage_booking_state tool.

    Args:
        arguments: Tool arguments with action and optional booking_id

    Returns:
        List of TextContent with booking information or confirmation
    """
    action = arguments.get('action')
    booking_id = arguments.get('booking_id')

    bookings_path = os.path.join('Data', 'bookings.json')
    try:
        with open(bookings_path, 'r', encoding='utf-8') as bf:
            bookings = json.load(bf)
    except Exception:
        bookings = []

    if action == 'get':
        if booking_id:
            # Get specific booking
            found = next((b for b in bookings if b['booking_id'] == booking_id), None)
            if not found:
                return [TextContent(type="text", text=f"❌ Booking {booking_id} not found")]
            text = data_helpers.format_booking_confirmation(found)
        else:
            # Get all bookings
            if not bookings:
                text = "No bookings found."
            else:
                text = f"Found {len(bookings)} booking(s):\n\n" + json.dumps(bookings, indent=2)

        return [TextContent(type="text", text=text)]

    elif action == 'cancel':
        if not booking_id:
            return [TextContent(type="text", text="❌ booking_id required to cancel")]

        idx = next((i for i, b in enumerate(bookings) if b['booking_id'] == booking_id), None)
        if idx is None:
            return [TextContent(type="text", text=f"❌ Booking {booking_id} not found")]

        # Mark as cancelled
        bookings[idx]['status'] = 'cancelled'

        # Return table to availability
        try:
            availability = data_helpers.load_availability(os.path.join('Data', 'availability.json'))
            r_id = bookings[idx]['restaurant_id']
            d = bookings[idx]['date']
            t = bookings[idx]['time']
            current_tables = availability.get(r_id, {}).get(d, {}).get(t, 0)
            availability.setdefault(r_id, {}).setdefault(d, {})[t] = current_tables + 1
            data_helpers.save_to_file(availability, os.path.join('Data', 'availability.json'))
        except Exception as e:
            pass

        # Save updated bookings
        data_helpers.save_to_file(bookings, bookings_path)

        text = f"✓ Booking {booking_id} has been cancelled successfully."
        return [TextContent(type="text", text=text)]

    else:
        return [TextContent(type="text", text=f"❌ Unknown action: {action}")]


# ============================================================================
# TOOL DISPATCHER
# ============================================================================

TOOL_HANDLERS = {
    "search_restaurants": handle_search_restaurants,
    "check_availability": handle_check_availability,
    "query_knowledge_base": handle_query_knowledge_base,
    "make_reservation": handle_make_reservation,
    "manage_booking_state": handle_manage_booking_state
}


async def execute_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Execute a tool by name with given arguments.

    Args:
        name: Tool name
        arguments: Tool arguments

    Returns:
        List of TextContent responses

    Raises:
        ValueError: If tool name is unknown
    """
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")

    return await handler(arguments)
