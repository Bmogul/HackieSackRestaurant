import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import random


def generate_availability(
    restaurant_ids: List[str],
    days_ahead: int = 14,
    save_to_file: bool = False
) -> str:
    """
    Generates availability data for restaurants in JSON format
    
    Args:
        restaurant_ids: List of restaurant IDs to generate availability for
        days_ahead: Number of days ahead to generate availability (default: 14)
        save_to_file: Whether to save the generated data to availability.json
        
    Returns:
        JSON string of generated availability data
    """
    availability = {}
    
    # Time slots for restaurant availability
    time_slots = [
        "11:30", "12:00", "12:30", "13:00", "13:30", "14:00",  # Lunch
        "17:00", "17:30", "18:00", "18:30", "19:00", "19:30",  # Early dinner
        "20:00", "20:30", "21:00", "21:30", "22:00"            # Late dinner
    ]
    
    # Get today's date
    today = datetime.now().date()
    
    # Generate availability for each restaurant
    for restaurant_id in restaurant_ids:
        availability[restaurant_id] = {}
        
        # Generate availability for each day
        for day_offset in range(days_ahead):
            date = today + timedelta(days=day_offset)
            date_string = date.strftime('%Y-%m-%d')
            
            # Day of week (0 = Monday, 6 = Sunday)
            day_of_week = date.weekday()
            
            availability[restaurant_id][date_string] = {}
            
            # Generate availability for each time slot
            for time_slot in time_slots:
                # Determine if this is lunch or dinner
                hour = int(time_slot.split(':')[0])
                is_lunch = hour < 15
                is_dinner = hour >= 17
                
                # Skip lunch slots on weekends for some restaurants (30% chance)
                if is_lunch and day_of_week >= 5 and random.random() < 0.3:
                    continue  # Skip this slot
                
                # Generate available tables based on various factors
                available_tables = 0
                
                # Base availability depends on time slot popularity
                if hour in [19, 20]:
                    # Peak dinner time (7-8pm) - lower availability
                    available_tables = random.randint(0, 2)  # 0-2 tables
                elif hour in [18, 21]:
                    # Popular times - moderate availability
                    available_tables = random.randint(0, 4)  # 0-4 tables
                elif is_lunch:
                    # Lunch - higher availability
                    available_tables = random.randint(2, 7)  # 2-7 tables
                else:
                    # Off-peak dinner - good availability
                    available_tables = random.randint(1, 8)  # 1-8 tables
                
                # Weekend dinner adjustment - busier
                if day_of_week >= 4 and is_dinner:  # Friday, Saturday, Sunday
                    available_tables = max(0, available_tables - 2)
                
                # Weekday lunch adjustment - busier
                if 0 <= day_of_week <= 4 and is_lunch:  # Monday to Friday
                    available_tables = max(0, available_tables - 1)
                
                # Random fully booked slots (20% chance for peak times)
                if hour in [19, 20] and random.random() < 0.2:
                    available_tables = 0
                
                availability[restaurant_id][date_string][time_slot] = available_tables
    
    json_string = json.dumps(availability, indent=2)
    
    if save_to_file:
        try:
            with open('availability.json', 'w', encoding='utf-8') as f:
                f.write(json_string)
            print(f"Successfully saved availability for {len(restaurant_ids)} restaurants to availability.json")
        except Exception as e:
            print(f"Error saving to file: {e}")
    
    return json_string


def generate_availability_from_file(
    restaurants_file_path: str = 'restaurants.json',
    days_ahead: int = 14,
    save_to_file: bool = False
) -> str:
    """
    Generates availability from restaurants.json file
    
    Args:
        restaurants_file_path: Path to restaurants.json file
        days_ahead: Number of days ahead to generate availability
        save_to_file: Whether to save the generated data to availability.json
        
    Returns:
        JSON string of generated availability data
    """
    try:
        with open(restaurants_file_path, 'r', encoding='utf-8') as f:
            restaurants = json.load(f)
        
        # Extract restaurant IDs
        restaurant_ids = [restaurant['id'] for restaurant in restaurants]
        
        print(f"Generating availability for {len(restaurant_ids)} restaurants...")
        
        return generate_availability(restaurant_ids, days_ahead, save_to_file)
    except Exception as e:
        print(f"Error reading restaurants file: {e}")
        raise


def check_availability(
    availability_data: Dict[str, Any],
    restaurant_id: str,
    date: str,
    time: str
) -> Optional[int]:
    """
    Check availability for a specific restaurant, date, and time
    
    Args:
        availability_data: Parsed availability JSON object
        restaurant_id: Restaurant ID
        date: Date in YYYY-MM-DD format
        time: Time in HH:MM format
        
    Returns:
        Number of available tables or None if not found
    """
    if restaurant_id not in availability_data:
        return None
    
    if date not in availability_data[restaurant_id]:
        return None
    
    return availability_data[restaurant_id][date].get(time)


def get_available_slots(
    availability_data: Dict[str, Any],
    restaurant_id: str,
    date: str,
    min_tables: int = 1
) -> List[Dict[str, Any]]:
    """
    Get all available time slots for a restaurant on a specific date
    
    Args:
        availability_data: Parsed availability JSON object
        restaurant_id: Restaurant ID
        date: Date in YYYY-MM-DD format
        min_tables: Minimum number of tables required (default: 1)
        
    Returns:
        List of available time slots with table counts
    """
    if restaurant_id not in availability_data or date not in availability_data[restaurant_id]:
        return []
    
    slots = availability_data[restaurant_id][date]
    available_slots = []
    
    for time, tables in slots.items():
        if tables >= min_tables:
            available_slots.append({
                'time': time,
                'available_tables': tables
            })
    
    return sorted(available_slots, key=lambda x: x['time'])


# Example usage
if __name__ == "__main__":
    # Example 1: Generate from restaurant IDs
    print("Example 1: Generating availability for specific restaurant IDs...\n")
    restaurant_ids = ['r1', 'r2', 'r3', 'r4', 'r5']
    result1 = generate_availability(restaurant_ids, days_ahead=7, save_to_file=False)
    print("Generated availability (first 500 chars):")
    print(result1[:500] + "...\n")
    
    # Example 2: Generate from restaurants.json file
    print("Example 2: Generating availability from restaurants.json file...\n")
    try:
        result2 = generate_availability_from_file('restaurants.json', days_ahead=14, save_to_file=True)
        print("Successfully generated and saved availability.json\n")
        
        # Example 3: Check specific availability
        availability_data = json.loads(result2)
        available = check_availability(availability_data, 'r1', '2024-01-20', '19:00')
        print(f"Example 3: Tables available at r1 on 2024-01-20 at 19:00: {available}\n")
        
        # Example 4: Get all available slots
        slots = get_available_slots(availability_data, 'r1', '2024-01-20', min_tables=1)
        print("Example 4: Available slots for r1 on 2024-01-20:")
        print(json.dumps(slots[:5], indent=2))
    except Exception as e:
        print(f"Could not read restaurants.json - make sure to generate it first using restaurant_generator.py")
        print(f"Error: {e}")
