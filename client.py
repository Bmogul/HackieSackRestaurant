"""
MCP Client for the restaurant booking server.

This client connects to the MCP server via stdio and uses the MCP protocol
to invoke tools exposed by `server.py`:

- search_restaurants
- check_availabilty
- query_knowledge_base
- make_reservation
- manage_booking_state

Usage examples:
    python client.py search --cuisine Italian --limit 5
    python client.py check --restaurant_id R001 --date 2025-11-03 --time 19:00
    python client.py query --query pizza
    python client.py reserve --restaurant_id R001 --date 2025-11-03 --time 19:00 --party_size 2 --customer_name "Alice" --customer_phone 555-0100
    python client.py manage get --booking_id BK20251102010101ABCD
"""

import argparse
import asyncio
import json
from typing import Optional, Any, Dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Call an MCP tool via the server."""
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "server"],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # Call the tool
            result = await session.call_tool(tool_name, arguments)

            # Extract text content from result
            if result.content:
                for content in result.content:
                    if hasattr(content, 'text'):
                        return content.text

            return "No response from server"


async def cmd_search(args: argparse.Namespace) -> None:
    """Search restaurants via MCP server."""
    arguments = {}
    if args.cuisine:
        arguments['cuisine'] = args.cuisine
    if args.location:
        arguments['location'] = args.location
    if args.price_range:
        arguments['price_range'] = args.price_range
    if args.dietary_options:
        arguments['dietary_options'] = args.dietary_options
    if args.min_rating is not None:
        arguments['min_rating'] = args.min_rating
    arguments['limit'] = args.limit or 10

    result = await call_mcp_tool("search_restaurants", arguments)
    print(result)


async def cmd_check(args: argparse.Namespace) -> None:
    """Check availability via MCP server."""
    arguments = {
        'restaurant_id': args.restaurant_id,
        'date': args.date,
        'time': args.time
    }

    result = await call_mcp_tool("check_availability", arguments)
    print(result)


async def cmd_query(args: argparse.Namespace) -> None:
    """Query knowledge base via MCP server."""
    arguments = {
        'query': args.query,
        'limit': args.limit or 5
    }

    result = await call_mcp_tool("query_knowledge_base", arguments)
    print(result)


async def cmd_reserve(args: argparse.Namespace) -> None:
    """Make a reservation via MCP server."""
    arguments = {
        'restaurant_id': args.restaurant_id,
        'date': args.date,
        'time': args.time,
        'party_size': args.party_size,
        'customer_name': args.customer_name,
        'customer_phone': args.customer_phone
    }

    if args.special_requests:
        arguments['special_requests'] = args.special_requests

    result = await call_mcp_tool("make_reservation", arguments)
    print(result)


async def cmd_manage(args: argparse.Namespace) -> None:
    """Manage bookings via MCP server."""
    arguments = {
        'action': args.action
    }

    if args.booking_id:
        arguments['booking_id'] = args.booking_id

    result = await call_mcp_tool("manage_booking_state", arguments)
    print(result)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(description='MCP Client for restaurant booking server')
    sub = parser.add_subparsers(dest='cmd')

    # search
    p_search = sub.add_parser('search', help='Search restaurants')
    p_search.add_argument('--cuisine', type=str, help='Filter by cuisine type')
    p_search.add_argument('--location', type=str, help='Filter by location')
    p_search.add_argument('--price_range', type=str, help='Filter by price range ($, $$, $$$, $$$$)')
    p_search.add_argument('--dietary_options', nargs='*', help='Filter by dietary options')
    p_search.add_argument('--min_rating', type=float, help='Minimum rating (0-5)')
    p_search.add_argument('--limit', type=int, default=10, help='Maximum number of results')
    p_search.set_defaults(func=cmd_search)

    # check
    p_check = sub.add_parser('check', help='Check availability')
    p_check.add_argument('--restaurant_id', required=True, help='Restaurant ID')
    p_check.add_argument('--date', required=True, help='Date (YYYY-MM-DD)')
    p_check.add_argument('--time', required=True, help='Time (HH:MM)')
    p_check.set_defaults(func=cmd_check)

    # query
    p_query = sub.add_parser('query', help='Query knowledge base')
    p_query.add_argument('--query', required=True, help='Search query')
    p_query.add_argument('--limit', type=int, default=5, help='Maximum number of results')
    p_query.set_defaults(func=cmd_query)

    # reserve
    p_res = sub.add_parser('reserve', help='Make a reservation')
    p_res.add_argument('--restaurant_id', required=True, help='Restaurant ID')
    p_res.add_argument('--date', required=True, help='Date (YYYY-MM-DD)')
    p_res.add_argument('--time', required=True, help='Time (HH:MM)')
    p_res.add_argument('--party_size', type=int, required=True, help='Number of guests')
    p_res.add_argument('--customer_name', required=True, help='Customer name')
    p_res.add_argument('--customer_phone', required=True, help='Customer phone')
    p_res.add_argument('--special_requests', help='Special requests')
    p_res.set_defaults(func=cmd_reserve)

    # manage
    p_m = sub.add_parser('manage', help='Manage bookings')
    p_m.add_argument('action', choices=['get', 'cancel'], help='Action to perform')
    p_m.add_argument('--booking_id', help='Booking ID')
    p_m.set_defaults(func=cmd_manage)

    return parser


async def async_main() -> None:
    """Async main function."""
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, 'func', None):
        parser.print_help()
        return

    # Call the async command function
    await args.func(args)


def main() -> None:
    """Main entry point."""
    asyncio.run(async_main())


if __name__ == '__main__':
    main()
