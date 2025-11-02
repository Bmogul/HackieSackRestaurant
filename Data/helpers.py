import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import random
import string


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Restaurant:
    id: str
    name: str
    cuisine: str
    location: str
    price_range: str
    rating: float
    dietary_options: List[str]
    hours: Dict[str, str]
    phone: str
    address: str
    features: List[str]
    popular_dishes: List[str]
    average_price_per_person: str
    reservations_required: bool
    outdoor_seating: bool
    private_dining: bool
    takeout_available: bool
    delivery_available: bool


@dataclass
class TimeSlot:
    time: str
    available_tables: int


@dataclass
class Booking:
    booking_id: str
    restaurant_id: str
    restaurant_name: str
    date: str
    time: str
    party_size: int
    customer_name: str
    customer_phone: str
    special_requests: Optional[str]
    status: str
    created_at: str


# ============================================================================
# FILE I/O HELPERS
# ============================================================================

def load_restaurants(file_path: str = 'restaurants.json') -> List[Dict[str, Any]]:
    """Load restaurants from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading restaurants from {file_path}: {e}")
        return []


def load_availability(file_path: str = 'availability.json') -> Dict[str, Any]:
    """Load availability from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading availability from {file_path}: {e}")
        return {}


def save_to_file(data: Any, file_path: str) -> bool:
    """Save data to JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")
        return False


# ============================================================================
# RESTAURANT SEARCH & FILTER HELPERS
# ============================================================================

def search_restaurants(
    restaurants: List[Dict[str, Any]],
    cuisine: Optional[str] = None,
    location: Optional[str] = None,
    price_range: Optional[str] = None,
    dietary_options: Optional[List[str]] = None,
    min_rating: Optional[float] = None,
    outdoor_seating: Optional[bool] = None,
    private_dining: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """Search restaurants by filters"""
    results = restaurants.copy()
    
    # Filter by cuisine
    if cuisine:
        results = [r for r in results if r['cuisine'].lower() == cuisine.lower()]
    
    # Filter by location
    if location:
        results = [r for r in results if r['location'].lower() == location.lower()]
    
    # Filter by price range
    if price_range:
        results = [r for r in results if r['price_range'] == price_range]
    
    # Filter by dietary options
    if dietary_options:
        results = [
            r for r in results
            if all(
                any(opt.lower() == ro.lower() for ro in r['dietary_options'])
                for opt in dietary_options
            )
        ]
    
    # Filter by minimum rating
    if min_rating is not None:
        results = [r for r in results if r['rating'] >= min_rating]
    
    # Filter by outdoor seating
    if outdoor_seating is not None:
        results = [r for r in results if r['outdoor_seating'] == outdoor_seating]
    
    # Filter by private dining
    if private_dining is not None:
        results = [r for r in results if r['private_dining'] == private_dining]
    
    return results


def get_restaurant_by_id(restaurants: List[Dict[str, Any]], restaurant_id: str) -> Optional[Dict[str, Any]]:
    """Get restaurant by ID"""
    for restaurant in restaurants:
        if restaurant['id'] == restaurant_id:
            return restaurant
    return None


def get_restaurants_by_cuisine(restaurants: List[Dict[str, Any]], cuisine: str) -> List[Dict[str, Any]]:
    """Get restaurants by cuisine type"""
    return [r for r in restaurants if r['cuisine'].lower() == cuisine.lower()]


def get_restaurants_by_location(restaurants: List[Dict[str, Any]], location: str) -> List[Dict[str, Any]]:
    """Get restaurants by location"""
    return [r for r in restaurants if r['location'].lower() == location.lower()]


def sort_by_rating(restaurants: List[Dict[str, Any]], descending: bool = True) -> List[Dict[str, Any]]:
    """Sort restaurants by rating"""
    return sorted(restaurants, key=lambda r: r['rating'], reverse=descending)


def sort_by_price(restaurants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort restaurants by price (ascending)"""
    price_order = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4}
    return sorted(restaurants, key=lambda r: price_order.get(r['price_range'], 2))


# ============================================================================
# AVAILABILITY HELPERS
# ============================================================================

def check_availability(
    availability: Dict[str, Any],
    restaurant_id: str,
    date: str,
    time: str
) -> Optional[int]:
    """Check if a restaurant has availability at a specific date and time"""
    try:
        return availability[restaurant_id][date][time]
    except KeyError:
        return None


def get_available_slots(
    availability: Dict[str, Any],
    restaurant_id: str,
    date: str,
    min_tables: int = 1
) -> List[Dict[str, Any]]:
    """Get all available time slots for a restaurant on a specific date"""
    try:
        date_slots = availability[restaurant_id][date]
        slots = [
            {'time': time, 'available_tables': tables}
            for time, tables in date_slots.items()
            if tables >= min_tables
        ]
        return sorted(slots, key=lambda x: x['time'])
    except KeyError:
        return []


def get_available_dates(
    availability: Dict[str, Any],
    restaurant_id: str,
    min_tables: int = 1
) -> List[str]:
    """Get available dates for a restaurant"""
    try:
        restaurant_availability = availability[restaurant_id]
        dates = [
            date for date, slots in restaurant_availability.items()
            if any(tables >= min_tables for tables in slots.values())
        ]
        return sorted(dates)
    except KeyError:
        return []


def find_alternative_slots(
    availability: Dict[str, Any],
    restaurant_id: str,
    date: str,
    preferred_time: str,
    min_tables: int = 1,
    max_alternatives: int = 3
) -> List[Dict[str, Any]]:
    """Find alternative time slots if preferred time is not available"""
    all_slots = get_available_slots(availability, restaurant_id, date, min_tables)
    
    # Remove preferred time
    alternatives = [slot for slot in all_slots if slot['time'] != preferred_time]
    
    # Sort by proximity to preferred time
    preferred_minutes = time_to_minutes(preferred_time)
    alternatives.sort(key=lambda slot: abs(time_to_minutes(slot['time']) - preferred_minutes))
    
    return alternatives[:max_alternatives]


def check_multiple_restaurants(
    availability: Dict[str, Any],
    restaurant_ids: List[str],
    date: str,
    time: str,
    min_tables: int = 1
) -> List[Dict[str, Any]]:
    """Check availability across multiple restaurants"""
    results = []
    for restaurant_id in restaurant_ids:
        tables = check_availability(availability, restaurant_id, date, time) or 0
        results.append({
            'restaurant_id': restaurant_id,
            'available': tables >= min_tables,
            'tables': tables
        })
    return results


# ============================================================================
# DATE & TIME HELPERS
# ============================================================================

def time_to_minutes(time: str) -> int:
    """Convert time string (HH:MM) to minutes since midnight"""
    hours, minutes = map(int, time.split(':'))
    return hours * 60 + minutes


def format_date(date: datetime) -> str:
    """Format date to YYYY-MM-DD"""
    return date.strftime('%Y-%m-%d')


def get_today() -> str:
    """Get today's date as YYYY-MM-DD"""
    return format_date(datetime.now())


def get_days_from_now(days: int) -> str:
    """Get date N days from today"""
    date = datetime.now() + timedelta(days=days)
    return format_date(date)


def parse_date(date_string: str) -> datetime:
    """Parse date string (YYYY-MM-DD) to datetime object"""
    return datetime.strptime(date_string, '%Y-%m-%d')


def is_date_in_past(date_string: str) -> bool:
    """Check if date is in the past"""
    date = parse_date(date_string)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return date < today


def get_day_of_week(date_string: str) -> str:
    """Get day of week name"""
    date = parse_date(date_string)
    return date.strftime('%A')


def is_weekend(date_string: str) -> bool:
    """Check if date is a weekend"""
    date = parse_date(date_string)
    return date.weekday() >= 5  # 5 = Saturday, 6 = Sunday


# ============================================================================
# BOOKING HELPERS
# ============================================================================

def generate_booking_id() -> str:
    """Generate a unique booking ID"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"BK{timestamp}{random_str}"


def create_booking(
    restaurant_id: str,
    restaurant_name: str,
    date: str,
    time: str,
    party_size: int,
    customer_name: str,
    customer_phone: str,
    special_requests: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new booking"""
    return {
        'booking_id': generate_booking_id(),
        'restaurant_id': restaurant_id,
        'restaurant_name': restaurant_name,
        'date': date,
        'time': time,
        'party_size': party_size,
        'customer_name': customer_name,
        'customer_phone': customer_phone,
        'special_requests': special_requests,
        'status': 'confirmed',
        'created_at': datetime.now().isoformat()
    }


def validate_booking(date: str, time: str, party_size: int) -> Tuple[bool, List[str]]:
    """Validate booking details"""
    errors = []
    
    # Check if date is in the past
    if is_date_in_past(date):
        errors.append('Cannot book for a past date')
    
    # Check party size
    if party_size < 1:
        errors.append('Party size must be at least 1')
    if party_size > 20:
        errors.append('Party size cannot exceed 20. Please contact restaurant directly for large groups')
    
    # Check time format
    try:
        hours, minutes = map(int, time.split(':'))
        if not (0 <= hours < 24 and 0 <= minutes < 60):
            errors.append('Invalid time format. Use HH:MM')
    except:
        errors.append('Invalid time format. Use HH:MM')
    
    return len(errors) == 0, errors


def update_availability_after_booking(
    availability: Dict[str, Any],
    restaurant_id: str,
    date: str,
    time: str,
    tables_booked: int = 1
) -> Dict[str, Any]:
    """Update availability after booking"""
    import copy
    updated = copy.deepcopy(availability)
    
    try:
        current = updated[restaurant_id][date][time]
        updated[restaurant_id][date][time] = max(0, current - tables_booked)
    except KeyError:
        pass
    
    return updated


# ============================================================================
# RECOMMENDATION HELPERS
# ============================================================================

def get_recommendations(
    restaurants: List[Dict[str, Any]],
    cuisine: Optional[str] = None,
    dietary_options: Optional[List[str]] = None,
    price_range: Optional[str] = None,
    min_rating: Optional[float] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Get restaurant recommendations based on preferences"""
    scored = []
    
    for restaurant in restaurants:
        score = 0
        
        # Exact cuisine match
        if cuisine and restaurant['cuisine'].lower() == cuisine.lower():
            score += 10
        
        # Dietary options match
        if dietary_options:
            match_count = sum(
                1 for option in dietary_options
                if any(opt.lower() == option.lower() for opt in restaurant['dietary_options'])
            )
            score += match_count * 5
        
        # Price range match
        if price_range and restaurant['price_range'] == price_range:
            score += 5
        
        # Rating bonus
        score += restaurant['rating'] * 2
        
        # Min rating filter
        if min_rating and restaurant['rating'] < min_rating:
            score = -1  # Exclude
        
        if score > 0:
            scored.append({'restaurant': restaurant, 'score': score})
    
    scored.sort(key=lambda x: x['score'], reverse=True)
    return [item['restaurant'] for item in scored[:limit]]


def find_similar_restaurants(
    restaurants: List[Dict[str, Any]],
    target_restaurant: Dict[str, Any],
    limit: int = 3
) -> List[Dict[str, Any]]:
    """Find similar restaurants"""
    scored = []
    
    for restaurant in restaurants:
        if restaurant['id'] == target_restaurant['id']:
            continue
        
        score = 0
        
        # Same cuisine
        if restaurant['cuisine'] == target_restaurant['cuisine']:
            score += 10
        
        # Similar price range
        if restaurant['price_range'] == target_restaurant['price_range']:
            score += 5
        
        # Same location
        if restaurant['location'] == target_restaurant['location']:
            score += 3
        
        # Similar rating (within 0.5)
        if abs(restaurant['rating'] - target_restaurant['rating']) <= 0.5:
            score += 2
        
        # Shared dietary options
        shared = len(set(restaurant['dietary_options']) & set(target_restaurant['dietary_options']))
        score += shared
        
        scored.append({'restaurant': restaurant, 'score': score})
    
    scored.sort(key=lambda x: x['score'], reverse=True)
    return [item['restaurant'] for item in scored[:limit]]


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def format_restaurant_display(restaurant: Dict[str, Any]) -> str:
    """Format restaurant for display"""
    return f"""
{restaurant['name']} ({restaurant['id']})
Cuisine: {restaurant['cuisine']}
Location: {restaurant['location']}
Price: {restaurant['price_range']} ({restaurant['average_price_per_person']})
Rating: {restaurant['rating']}/5.0
Dietary Options: {', '.join(restaurant['dietary_options'])}
Phone: {restaurant['phone']}
Address: {restaurant['address']}
Features: {', '.join(restaurant['features'])}
Popular Dishes: {', '.join(restaurant['popular_dishes'])}
    """.strip()


def format_booking_confirmation(booking: Dict[str, Any]) -> str:
    """Format booking confirmation"""
    special_requests = f"\nSpecial Requests: {booking['special_requests']}" if booking.get('special_requests') else ""
    
    return f"""
BOOKING CONFIRMATION
Confirmation Number: {booking['booking_id']}
Restaurant: {booking['restaurant_name']}
Date: {booking['date']} ({get_day_of_week(booking['date'])})
Time: {booking['time']}
Party Size: {booking['party_size']}
Name: {booking['customer_name']}
Phone: {booking['customer_phone']}{special_requests}
Status: {booking['status'].upper()}
Booked on: {booking['created_at']}
    """.strip()


def format_price_range(price_symbol: str) -> str:
    """Format price range to readable string"""
    ranges = {
        '$': 'Budget-friendly ($10-20)',
        '$$': 'Moderate ($20-40)',
        '$$$': 'Upscale ($40-70)',
        '$$$$': 'Fine Dining ($70-150)'
    }
    return ranges.get(price_symbol, price_symbol)


# ============================================================================
# STATISTICS HELPERS
# ============================================================================

def get_restaurant_stats(restaurants: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get restaurant statistics"""
    stats = {
        'total': len(restaurants),
        'by_cuisine': {},
        'by_location': {},
        'by_price_range': {},
        'average_rating': 0
    }
    
    total_rating = 0
    
    for restaurant in restaurants:
        # Count by cuisine
        cuisine = restaurant['cuisine']
        stats['by_cuisine'][cuisine] = stats['by_cuisine'].get(cuisine, 0) + 1
        
        # Count by location
        location = restaurant['location']
        stats['by_location'][location] = stats['by_location'].get(location, 0) + 1
        
        # Count by price range
        price = restaurant['price_range']
        stats['by_price_range'][price] = stats['by_price_range'].get(price, 0) + 1
        
        # Sum ratings
        total_rating += restaurant['rating']
    
    stats['average_rating'] = total_rating / len(restaurants) if restaurants else 0
    
    return stats


def get_availability_stats(availability: Dict[str, Any], restaurant_id: str) -> Dict[str, Any]:
    """Get availability statistics for a restaurant"""
    try:
        restaurant_availability = availability[restaurant_id]
    except KeyError:
        return {
            'total_slots': 0,
            'available_slots': 0,
            'fully_booked_slots': 0,
            'availability_rate': 0
        }
    
    total_slots = 0
    available_slots = 0
    fully_booked_slots = 0
    
    for date_slots in restaurant_availability.values():
        for tables in date_slots.values():
            total_slots += 1
            if tables > 0:
                available_slots += 1
            else:
                fully_booked_slots += 1
    
    return {
        'total_slots': total_slots,
        'available_slots': available_slots,
        'fully_booked_slots': fully_booked_slots,
        'availability_rate': (available_slots / total_slots * 100) if total_slots > 0 else 0
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=== Restaurant Booking Helper Functions ===\n")
    
    # Load data
    print("1. Loading data...")
    restaurants = load_restaurants()
    availability = load_availability()
    print(f"   Loaded {len(restaurants)} restaurants\n")
    
    # Search restaurants
    print("2. Searching for Italian restaurants...")
    italian = search_restaurants(restaurants, cuisine="Italian")
    print(f"   Found {len(italian)} Italian restaurants\n")
    
    # Check availability
    if restaurants and availability:
        print("3. Checking availability...")
        restaurant_id = restaurants[0]['id']
        today = get_today()
        tables = check_availability(availability, restaurant_id, today, "19:00")
        print(f"   Restaurant {restaurant_id} has {tables} tables at 19:00 today\n")
    
    # Get recommendations
    print("4. Getting recommendations...")
    recommendations = get_recommendations(
        restaurants,
        cuisine="Italian",
        dietary_options=["vegetarian"],
        min_rating=4.0,
        limit=3
    )
    print(f"   Found {len(recommendations)} recommendations\n")
    
    # Create booking
    print("5. Creating sample booking...")
    if restaurants:
        booking = create_booking(
            restaurant_id=restaurants[0]['id'],
            restaurant_name=restaurants[0]['name'],
            date=get_days_from_now(1),
            time="19:00",
            party_size=4,
            customer_name="John Doe",
            customer_phone="555-1234"
        )
        print(f"   Created booking: {booking['booking_id']}\n")
    
    # Get statistics
    print("6. Restaurant statistics...")
    stats = get_restaurant_stats(restaurants)
    print(f"   Total restaurants: {stats['total']}")
    print(f"   Average rating: {stats['average_rating']:.2f}")
    print(f"   Cuisines: {', '.join(stats['by_cuisine'].keys())}")
