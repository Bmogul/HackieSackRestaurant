"""
Simple CLI client for the MCP restaurant server.

This client can operate in "direct" mode by calling the local Data.helpers
functions (useful for local testing). It provides commands that correspond to
the MCP tools exposed by `server.py`:

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

Note: This is a lightweight local client that calls the same helpers used by
`server.py`. It does not implement the MCP stdio protocol. If you want an MCP
client that connects over stdio, we can add that next (it needs the `mcp`
library and a small JSON-RPC framing implementation).
"""

import argparse
import json
import os
from typing import Optional

from Data import helpers as data_helpers

DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data')
RESTAURANTS_PATH = os.path.join(DATA_DIR, 'restaurants.json')
AVAILABILITY_PATH = os.path.join(DATA_DIR, 'availability.json')
BOOKINGS_PATH = os.path.join(DATA_DIR, 'bookings.json')


def cmd_search(args: argparse.Namespace) -> None:
    restaurants = data_helpers.load_restaurants(RESTAURANTS_PATH)
    results = data_helpers.search_restaurants(
        restaurants,
        cuisine=args.cuisine,
        location=args.location,
        price_range=args.price_range,
        dietary_options=args.dietary_options,
        min_rating=args.min_rating
    )

    for r in results[: args.limit or 10]:
        print(data_helpers.format_restaurant_display(r))
        print('-' * 60)


def cmd_check(args: argparse.Namespace) -> None:
    availability = data_helpers.load_availability(AVAILABILITY_PATH)
    tables = data_helpers.check_availability(availability, args.restaurant_id, args.date, args.time)
    if tables is None:
        print(f"No availability data for restaurant {args.restaurant_id} at {args.date} {args.time}")
    else:
        print(f"Restaurant {args.restaurant_id} has {tables} table(s) available at {args.date} {args.time}")


def cmd_query(args: argparse.Namespace) -> None:
    restaurants = data_helpers.load_restaurants(RESTAURANTS_PATH)
    q = (args.query or '').lower()
    matches = []
    for r in restaurants:
        hay = ' '.join([r.get('name',''), r.get('cuisine',''), ' '.join(r.get('features',[])), ' '.join(r.get('popular_dishes',[]))]).lower()
        if q in hay:
            matches.append(r)

    if not matches:
        print('No knowledge-base matches found.')
        return

    for m in matches[: args.limit or 5]:
        print(data_helpers.format_restaurant_display(m))
        print('-' * 60)


def cmd_reserve(args: argparse.Namespace) -> None:
    # Validate
    ok, errors = data_helpers.validate_booking(args.date, args.time, args.party_size)
    if not ok:
        print('Booking validation failed:', errors)
        return

    restaurants = data_helpers.load_restaurants(RESTAURANTS_PATH)
    restaurant = data_helpers.get_restaurant_by_id(restaurants, args.restaurant_id)
    if not restaurant:
        print('Restaurant not found:', args.restaurant_id)
        return

    availability = data_helpers.load_availability(AVAILABILITY_PATH)
    tables = data_helpers.check_availability(availability, args.restaurant_id, args.date, args.time) or 0
    if tables < 1:
        alternatives = data_helpers.find_alternative_slots(availability, args.restaurant_id, args.date, args.time)
        if not alternatives:
            print('No availability and no alternatives found.')
        else:
            print('No tables available. Alternatives:')
            for a in alternatives:
                print(f"  {a['time']} ({a['available_tables']})")
        return

    booking = data_helpers.create_booking(
        restaurant_id=args.restaurant_id,
        restaurant_name=restaurant['name'],
        date=args.date,
        time=args.time,
        party_size=args.party_size,
        customer_name=args.customer_name,
        customer_phone=args.customer_phone,
        special_requests=args.special_requests
    )

    # Persist booking
    try:
        with open(BOOKINGS_PATH, 'r', encoding='utf-8') as bf:
            bookings = json.load(bf)
    except Exception:
        bookings = []

    bookings.append(booking)
    data_helpers.save_to_file(bookings, BOOKINGS_PATH)

    # Update availability
    updated = data_helpers.update_availability_after_booking(availability, args.restaurant_id, args.date, args.time, tables_booked=1)
    data_helpers.save_to_file(updated, AVAILABILITY_PATH)

    print(data_helpers.format_booking_confirmation(booking))


def cmd_manage(args: argparse.Namespace) -> None:
    try:
        with open(BOOKINGS_PATH, 'r', encoding='utf-8') as bf:
            bookings = json.load(bf)
    except Exception:
        bookings = []

    if args.action == 'get':
        if args.booking_id:
            found = next((b for b in bookings if b['booking_id'] == args.booking_id), None)
            if not found:
                print('Booking not found:', args.booking_id)
            else:
                print(data_helpers.format_booking_confirmation(found))
        else:
            print(json.dumps(bookings, indent=2))

    elif args.action == 'cancel':
        if not args.booking_id:
            print('booking_id required to cancel')
            return
        idx = next((i for i,b in enumerate(bookings) if b['booking_id']==args.booking_id), None)
        if idx is None:
            print('Booking not found:', args.booking_id)
            return

        bookings[idx]['status'] = 'cancelled'

        # Return a table to availability (naive +1)
        try:
            availability = data_helpers.load_availability(AVAILABILITY_PATH)
            r_id = bookings[idx]['restaurant_id']
            d = bookings[idx]['date']
            t = bookings[idx]['time']
            availability.setdefault(r_id, {}).setdefault(d, {})[t] = availability.get(r_id, {}).get(d, {}).get(t, 0) + 1
            data_helpers.save_to_file(availability, AVAILABILITY_PATH)
        except Exception:
            pass

        data_helpers.save_to_file(bookings, BOOKINGS_PATH)
        print('Booking cancelled:', args.booking_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Client for MCP restaurant server (local helper mode)')
    sub = parser.add_subparsers(dest='cmd')

    # search
    p_search = sub.add_parser('search', help='Search restaurants')
    p_search.add_argument('--cuisine', type=str)
    p_search.add_argument('--location', type=str)
    p_search.add_argument('--price_range', type=str)
    p_search.add_argument('--dietary_options', nargs='*')
    p_search.add_argument('--min_rating', type=float)
    p_search.add_argument('--limit', type=int, default=10)
    p_search.set_defaults(func=cmd_search)

    # check
    p_check = sub.add_parser('check', help='Check availability')
    p_check.add_argument('--restaurant_id', required=True)
    p_check.add_argument('--date', required=True)
    p_check.add_argument('--time', required=True)
    p_check.set_defaults(func=cmd_check)

    # query
    p_query = sub.add_parser('query', help='Query knowledge base')
    p_query.add_argument('--query', required=True)
    p_query.add_argument('--limit', type=int, default=5)
    p_query.set_defaults(func=cmd_query)

    # reserve
    p_res = sub.add_parser('reserve', help='Make a reservation')
    p_res.add_argument('--restaurant_id', required=True)
    p_res.add_argument('--date', required=True)
    p_res.add_argument('--time', required=True)
    p_res.add_argument('--party_size', type=int, required=True)
    p_res.add_argument('--customer_name', required=True)
    p_res.add_argument('--customer_phone', required=True)
    p_res.add_argument('--special_requests')
    p_res.set_defaults(func=cmd_reserve)

    # manage
    p_m = sub.add_parser('manage', help='Manage bookings')
    p_m.add_argument('action', choices=['get', 'cancel'])
    p_m.add_argument('--booking_id')
    p_m.set_defaults(func=cmd_manage)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, 'func', None):
        parser.print_help()
        return
    args.func(args)


if __name__ == '__main__':
    main()
