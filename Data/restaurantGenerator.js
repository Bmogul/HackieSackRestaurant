const fs = require('fs');

/**
 * Generates restaurant data in JSON format
 * @param {number} count - Number of restaurant objects to generate
 * @param {boolean} saveToFile - Whether to save the generated data to restaurants.json
 * @returns {string} JSON string of generated restaurants
 */
function generateRestaurants(count, saveToFile = false) {
  const restaurants = [];
  
  const restaurantTemplates = [
    {
      namePrefix: "Bella",
      nameSuffix: ["Italia", "Napoli", "Roma", "Toscana", "Venezia"],
      cuisine: "Italian",
      location: ["Downtown", "Midtown", "Uptown", "Old Town", "West End"],
      price_range: "$$$",
      dietary_base: ["vegetarian", "gluten-free", "vegan"],
      features_base: ["romantic ambiance", "wine bar", "pasta made fresh daily", "outdoor patio"],
      dishes_base: ["Truffle Carbonara", "Osso Buco", "Margherita Pizza", "Tiramisu"]
    },
    {
      namePrefix: "Sakura",
      nameSuffix: ["Sushi", "Japanese Bistro", "Izakaya", "Ramen House", "Sushi Bar"],
      cuisine: "Japanese",
      location: ["Downtown", "Midtown", "Uptown", "East Side", "Arts District"],
      price_range: "$$",
      dietary_base: ["vegetarian", "gluten-free", "vegan"],
      features_base: ["sushi bar seating", "sake selection", "omakase available", "authentic Japanese"],
      dishes_base: ["Omakase", "Spicy Tuna Roll", "Ramen", "Tempura"]
    },
    {
      namePrefix: "The Steakhouse",
      nameSuffix: ["Prime", "Grill", "Chophouse", "& Co", "Club"],
      cuisine: "American Steakhouse",
      location: ["Financial District", "Downtown", "Uptown", "Business District", "Harbor"],
      price_range: "$$$$",
      dietary_base: ["gluten-free"],
      features_base: ["dry-aged beef", "wine cellar", "private dining rooms", "valet parking"],
      dishes_base: ["Ribeye Steak", "Filet Mignon", "Lobster Tail", "NY Cheesecake"]
    },
    {
      namePrefix: "Spice",
      nameSuffix: ["of India", "Kitchen", "Palace", "Garden", "Tandoor"],
      cuisine: "Indian",
      location: ["Midtown", "University District", "Downtown", "West End", "Little India"],
      price_range: "$$",
      dietary_base: ["vegetarian", "vegan", "gluten-free"],
      features_base: ["tandoor oven", "lunch buffet", "authentic spices", "family-owned"],
      dishes_base: ["Chicken Tikka Masala", "Palak Paneer", "Biryani", "Naan Bread"]
    },
    {
      namePrefix: "Le",
      nameSuffix: ["Bistro", "Café", "Jardin", "Petit", "Bouchon"],
      cuisine: "French",
      location: ["Downtown", "Arts District", "Old Town", "Riverside", "Historic District"],
      price_range: "$$$",
      dietary_base: ["vegetarian", "gluten-free"],
      features_base: ["french wine list", "outdoor seating", "romantic setting", "chef-owned"],
      dishes_base: ["Coq au Vin", "Bouillabaisse", "Crème Brûlée", "Escargot"]
    },
    {
      namePrefix: "Taco",
      nameSuffix: ["Loco", "Fiesta", "Cantina", "Casa", "Express"],
      cuisine: "Mexican",
      location: ["Downtown", "Beach Area", "Midtown", "South Side", "Market District"],
      price_range: "$",
      dietary_base: ["vegetarian", "vegan", "gluten-free"],
      features_base: ["margarita bar", "taco tuesday", "fresh ingredients", "casual dining"],
      dishes_base: ["Street Tacos", "Carnitas", "Guacamole", "Churros"]
    },
    {
      namePrefix: "Dragon",
      nameSuffix: ["Palace", "Garden", "Wok", "House", "Kitchen"],
      cuisine: "Chinese",
      location: ["Chinatown", "Downtown", "Midtown", "East Side", "University Area"],
      price_range: "$$",
      dietary_base: ["vegetarian", "vegan", "gluten-free"],
      features_base: ["dim sum", "family-style dining", "authentic recipes", "lunch specials"],
      dishes_base: ["Peking Duck", "Kung Pao Chicken", "Dumplings", "Fried Rice"]
    },
    {
      namePrefix: "Mediterranean",
      nameSuffix: ["Grill", "Kitchen", "Taverna", "Cafe", "Bistro"],
      cuisine: "Mediterranean",
      location: ["Downtown", "Waterfront", "Old Town", "Arts District", "Beach Area"],
      price_range: "$$",
      dietary_base: ["vegetarian", "vegan", "gluten-free"],
      features_base: ["healthy options", "fresh seafood", "outdoor patio", "mezze platters"],
      dishes_base: ["Lamb Kebab", "Falafel", "Hummus Platter", "Baklava"]
    }
  ];

  for (let i = 0; i < count; i++) {
    const template = restaurantTemplates[i % restaurantTemplates.length];
    const suffixIndex = Math.floor(i / restaurantTemplates.length) % template.nameSuffix.length;
    const locationIndex = i % template.location.length;
    
    const restaurant = {
      id: `r${i + 1}`,
      name: `${template.namePrefix} ${template.nameSuffix[suffixIndex]}`,
      cuisine: template.cuisine,
      location: template.location[locationIndex],
      price_range: template.price_range,
      rating: parseFloat((4.0 + Math.random() * 1.0).toFixed(1)),
      dietary_options: template.dietary_base,
      hours: {
        dinner: "5:00pm-10:00pm"
      },
      phone: `(555) ${String(Math.floor(100 + Math.random() * 900)).padStart(3, '0')}-${String(Math.floor(1000 + Math.random() * 9000)).padStart(4, '0')}`,
      address: `${Math.floor(100 + Math.random() * 900)} ${template.location[locationIndex]} Street`,
      features: template.features_base,
      popular_dishes: template.dishes_base,
      average_price_per_person: getPriceRange(template.price_range),
      reservations_required: template.price_range.length >= 3,
      outdoor_seating: Math.random() > 0.5,
      private_dining: template.price_range.length >= 3,
      takeout_available: true,
      delivery_available: Math.random() > 0.3
    };

    // Add lunch hours for 70% of restaurants
    if (Math.random() > 0.3) {
      restaurant.hours.lunch = "11:30am-2:30pm";
    }

    restaurants.push(restaurant);
  }

  const jsonString = JSON.stringify(restaurants, null, 2);

  if (saveToFile) {
    try {
      fs.writeFileSync('restaurants.json', jsonString, 'utf-8');
      console.log(`Successfully saved ${count} restaurants to restaurants.json`);
    } catch (error) {
      console.error('Error saving to file:', error);
    }
  }

  return jsonString;
}

/**
 * Convert price symbols to price ranges
 * @param {string} priceSymbol - Price range symbol ($, $$, $$$, $$$$)
 * @returns {string} Price range string
 */
function getPriceRange(priceSymbol) {
  const ranges = {
    '$': '$10-20',
    '$$': '$20-40',
    '$$$': '$40-70',
    '$$$$': '$70-150'
  };
  return ranges[priceSymbol] || '$20-40';
}

// Export for use as module
module.exports = {
  generateRestaurants,
  getPriceRange
};

// Example usage when run directly
if (require.main === module) {
  // Generate 5 restaurants and save to file
  const result = generateRestaurants(5, true);
  console.log('\nGenerated restaurants:');
  console.log(result);
}
