import json
import random
from typing import List, Dict, Any, Optional


def generate_restaurants(count: int, save_to_file: bool = False) -> str:
    """
    Generates restaurant data in JSON format
    
    Args:
        count: Number of restaurant objects to generate
        save_to_file: Whether to save the generated data to restaurants.json
        
    Returns:
        JSON string of generated restaurants
    """
    restaurants: List[Dict[str, Any]] = []
    
    restaurant_templates = [
        {
            "name_prefix": "Bella",
            "name_suffix": ["Italia", "Napoli", "Roma", "Toscana", "Venezia"],
            "cuisine": "Italian",
            "location": ["Downtown", "Midtown", "Uptown", "Old Town", "West End"],
            "price_range": "$$$",
            "dietary_base": ["vegetarian", "gluten-free", "vegan"],
            "features_base": ["romantic ambiance", "wine bar", "pasta made fresh daily", "outdoor patio"],
            "dishes_base": ["Truffle Carbonara", "Osso Buco", "Margherita Pizza", "Tiramisu"]
        },
        {
            "name_prefix": "Sakura",
            "name_suffix": ["Sushi", "Japanese Bistro", "Izakaya", "Ramen House", "Sushi Bar"],
            "cuisine": "Japanese",
            "location": ["Downtown", "Midtown", "Uptown", "East Side", "Arts District"],
            "price_range": "$$",
            "dietary_base": ["vegetarian", "gluten-free", "vegan"],
            "features_base": ["sushi bar seating", "sake selection", "omakase available", "authentic Japanese"],
            "dishes_base": ["Omakase", "Spicy Tuna Roll", "Ramen", "Tempura"]
        },
        {
            "name_prefix": "The Steakhouse",
            "name_suffix": ["Prime", "Grill", "Chophouse", "& Co", "Club"],
            "cuisine": "American Steakhouse",
            "location": ["Financial District", "Downtown", "Uptown", "Business District", "Harbor"],
            "price_range": "$$$$",
            "dietary_base": ["gluten-free"],
            "features_base": ["dry-aged beef", "wine cellar", "private dining rooms", "valet parking"],
            "dishes_base": ["Ribeye Steak", "Filet Mignon", "Lobster Tail", "NY Cheesecake"]
        },
        {
            "name_prefix": "Spice",
            "name_suffix": ["of India", "Kitchen", "Palace", "Garden", "Tandoor"],
            "cuisine": "Indian",
            "location": ["Midtown", "University District", "Downtown", "West End", "Little India"],
            "price_range": "$$",
            "dietary_base": ["vegetarian", "vegan", "gluten-free"],
            "features_base": ["tandoor oven", "lunch buffet", "authentic spices", "family-owned"],
            "dishes_base": ["Chicken Tikka Masala", "Palak Paneer", "Biryani", "Naan Bread"]
        },
        {
            "name_prefix": "Le",
            "name_suffix": ["Bistro", "Café", "Jardin", "Petit", "Bouchon"],
            "cuisine": "French",
            "location": ["Downtown", "Arts District", "Old Town", "Riverside", "Historic District"],
            "price_range": "$$$",
            "dietary_base": ["vegetarian", "gluten-free"],
            "features_base": ["french wine list", "outdoor seating", "romantic setting", "chef-owned"],
            "dishes_base": ["Coq au Vin", "Bouillabaisse", "Crème Brûlée", "Escargot"]
        },
        {
            "name_prefix": "Taco",
            "name_suffix": ["Loco", "Fiesta", "Cantina", "Casa", "Express"],
            "cuisine": "Mexican",
            "location": ["Downtown", "Beach Area", "Midtown", "South Side", "Market District"],
            "price_range": "$",
            "dietary_base": ["vegetarian", "vegan", "gluten-free"],
            "features_base": ["margarita bar", "taco tuesday", "fresh ingredients", "casual dining"],
            "dishes_base": ["Street Tacos", "Carnitas", "Guacamole", "Churros"]
        },
        {
            "name_prefix": "Dragon",
            "name_suffix": ["Palace", "Garden", "Wok", "House", "Kitchen"],
            "cuisine": "Chinese",
            "location": ["Chinatown", "Downtown", "Midtown", "East Side", "University Area"],
            "price_range": "$$",
            "dietary_base": ["vegetarian", "vegan", "gluten-free"],
            "features_base": ["dim sum", "family-style dining", "authentic recipes", "lunch specials"],
            "dishes_base": ["Peking Duck", "Kung Pao Chicken", "Dumplings", "Fried Rice"]
        },
        {
            "name_prefix": "Mediterranean",
            "name_suffix": ["Grill", "Kitchen", "Taverna", "Cafe", "Bistro"],
            "cuisine": "Mediterranean",
            "location": ["Downtown", "Waterfront", "Old Town", "Arts District", "Beach Area"],
            "price_range": "$$",
            "dietary_base": ["vegetarian", "vegan", "gluten-free"],
            "features_base": ["healthy options", "fresh seafood", "outdoor patio", "mezze platters"],
            "dishes_base": ["Lamb Kebab", "Falafel", "Hummus Platter", "Baklava"]
        }
    ]
    
    for i in range(count):
        template = restaurant_templates[i % len(restaurant_templates)]
        suffix_index = (i // len(restaurant_templates)) % len(template["name_suffix"])
        location_index = i % len(template["location"])
        
        restaurant = {
            "id": f"r{i + 1}",
            "name": f"{template['name_prefix']} {template['name_suffix'][suffix_index]}",
            "cuisine": template["cuisine"],
            "location": template["location"][location_index],
            "price_range": template["price_range"],
            "rating": round(4.0 + random.random() * 1.0, 1),
            "dietary_options": template["dietary_base"],
            "hours": {
                "dinner": "5:00pm-10:00pm"
            },
            "phone": f"(555) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
            "address": f"{random.randint(100, 999)} {template['location'][location_index]} Street",
            "features": template["features_base"],
            "popular_dishes": template["dishes_base"],
            "average_price_per_person": get_price_range(template["price_range"]),
            "reservations_required": len(template["price_range"]) >= 3,
            "outdoor_seating": random.random() > 0.5,
            "private_dining": len(template["price_range"]) >= 3,
            "takeout_available": True,
            "delivery_available": random.random() > 0.3
        }
        
        # Add lunch hours for 70% of restaurants
        if random.random() > 0.3:
            restaurant["hours"]["lunch"] = "11:30am-2:30pm"
        
        restaurants.append(restaurant)
    
    json_string = json.dumps(restaurants, indent=2)
    
    if save_to_file:
        try:
            with open('restaurants.json', 'w', encoding='utf-8') as f:
                f.write(json_string)
            print(f"Successfully saved {count} restaurants to restaurants.json")
        except Exception as e:
            print(f"Error saving to file: {e}")
    
    return json_string


def get_price_range(price_symbol: str) -> str:
    """Convert price symbols to price ranges"""
    ranges = {
        '$': '$10-20',
        '$$': '$20-40',
        '$$$': '$40-70',
        '$$$$': '$70-150'
    }
    return ranges.get(price_symbol, '$20-40')


# Example usage
if __name__ == "__main__":
    # Generate 5 restaurants and save to file
    result = generate_restaurants(5, save_to_file=True)
    print("\nGenerated restaurants:")
    print(result)
