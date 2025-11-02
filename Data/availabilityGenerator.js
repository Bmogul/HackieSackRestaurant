const fs = require('fs');

/**
 * Generates availability data for restaurants in JSON format
 * @param {Array} restaurantIds - Array of restaurant IDs to generate availability for
 * @param {number} daysAhead - Number of days ahead to generate availability (default: 14)
 * @param {boolean} saveToFile - Whether to save the generated data to availability.json
 * @returns {string} JSON string of generated availability data
 */
function generateAvailability(restaurantIds, daysAhead = 14, saveToFile = false) {
  const availability = {};
  
  // Time slots for restaurant availability
  const timeSlots = [
    "11:30", "12:00", "12:30", "13:00", "13:30", "14:00",  // Lunch
    "17:00", "17:30", "18:00", "18:30", "19:00", "19:30",  // Early dinner
    "20:00", "20:30", "21:00", "21:30", "22:00"            // Late dinner
  ];
  
  // Get today's date
  const today = new Date();
  
  // Generate availability for each restaurant
  restaurantIds.forEach(restaurantId => {
    availability[restaurantId] = {};
    
    // Generate availability for each day
    for (let dayOffset = 0; dayOffset < daysAhead; dayOffset++) {
      const date = new Date(today);
      date.setDate(today.getDate() + dayOffset);
      
      // Format date as YYYY-MM-DD
      const dateString = date.toISOString().split('T')[0];
      
      // Day of week (0 = Sunday, 6 = Saturday)
      const dayOfWeek = date.getDay();
      
      availability[restaurantId][dateString] = {};
      
      // Generate availability for each time slot
      timeSlots.forEach(timeSlot => {
        // Determine if this is lunch or dinner
        const hour = parseInt(timeSlot.split(':')[0]);
        const isLunch = hour < 15;
        const isDinner = hour >= 17;
        
        // Skip lunch slots on weekends for some restaurants (30% chance)
        if (isLunch && (dayOfWeek === 0 || dayOfWeek === 6) && Math.random() < 0.3) {
          return; // Skip this slot
        }
        
        // Generate available tables based on various factors
        let availableTables = 0;
        
        // Base availability depends on time slot popularity
        if (hour === 19 || hour === 20) {
          // Peak dinner time (7-8pm) - lower availability
          availableTables = Math.floor(Math.random() * 3); // 0-2 tables
        } else if (hour === 18 || hour === 21) {
          // Popular times - moderate availability
          availableTables = Math.floor(Math.random() * 5); // 0-4 tables
        } else if (isLunch) {
          // Lunch - higher availability
          availableTables = Math.floor(Math.random() * 6) + 2; // 2-7 tables
        } else {
          // Off-peak dinner - good availability
          availableTables = Math.floor(Math.random() * 8) + 1; // 1-8 tables
        }
        
        // Weekend dinner adjustment - busier
        if ((dayOfWeek === 5 || dayOfWeek === 6) && isDinner) {
          availableTables = Math.max(0, availableTables - 2);
        }
        
        // Weekday lunch adjustment - busier
        if (dayOfWeek >= 1 && dayOfWeek <= 5 && isLunch) {
          availableTables = Math.max(0, availableTables - 1);
        }
        
        // Random fully booked slots (20% chance for peak times)
        if ((hour === 19 || hour === 20) && Math.random() < 0.2) {
          availableTables = 0;
        }
        
        availability[restaurantId][dateString][timeSlot] = availableTables;
      });
    }
  });
  
  const jsonString = JSON.stringify(availability, null, 2);
  
  if (saveToFile) {
    try {
      fs.writeFileSync('availability.json', jsonString, 'utf-8');
      console.log(`Successfully saved availability for ${restaurantIds.length} restaurants to availability.json`);
    } catch (error) {
      console.error('Error saving to file:', error);
    }
  }
  
  return jsonString;
}

/**
 * Generates availability from restaurants.json file
 * @param {string} restaurantsFilePath - Path to restaurants.json file
 * @param {number} daysAhead - Number of days ahead to generate availability
 * @param {boolean} saveToFile - Whether to save the generated data to availability.json
 * @returns {string} JSON string of generated availability data
 */
function generateAvailabilityFromFile(restaurantsFilePath = 'restaurants.json', daysAhead = 14, saveToFile = false) {
  try {
    const restaurantsData = fs.readFileSync(restaurantsFilePath, 'utf-8');
    const restaurants = JSON.parse(restaurantsData);
    
    // Extract restaurant IDs
    const restaurantIds = restaurants.map(restaurant => restaurant.id);
    
    console.log(`Generating availability for ${restaurantIds.length} restaurants...`);
    
    return generateAvailability(restaurantIds, daysAhead, saveToFile);
  } catch (error) {
    console.error('Error reading restaurants file:', error);
    throw error;
  }
}

/**
 * Check availability for a specific restaurant, date, and time
 * @param {Object} availabilityData - Parsed availability JSON object
 * @param {string} restaurantId - Restaurant ID
 * @param {string} date - Date in YYYY-MM-DD format
 * @param {string} time - Time in HH:MM format
 * @returns {number|null} Number of available tables or null if not found
 */
function checkAvailability(availabilityData, restaurantId, date, time) {
  if (!availabilityData[restaurantId]) {
    return null;
  }
  
  if (!availabilityData[restaurantId][date]) {
    return null;
  }
  
  return availabilityData[restaurantId][date][time] ?? null;
}

/**
 * Get all available time slots for a restaurant on a specific date
 * @param {Object} availabilityData - Parsed availability JSON object
 * @param {string} restaurantId - Restaurant ID
 * @param {string} date - Date in YYYY-MM-DD format
 * @param {number} minTables - Minimum number of tables required (default: 1)
 * @returns {Array} Array of available time slots with table counts
 */
function getAvailableSlots(availabilityData, restaurantId, date, minTables = 1) {
  if (!availabilityData[restaurantId] || !availabilityData[restaurantId][date]) {
    return [];
  }
  
  const slots = availabilityData[restaurantId][date];
  const availableSlots = [];
  
  for (const [time, tables] of Object.entries(slots)) {
    if (tables >= minTables) {
      availableSlots.push({
        time: time,
        available_tables: tables
      });
    }
  }
  
  return availableSlots.sort((a, b) => a.time.localeCompare(b.time));
}

// Export functions
module.exports = {
  generateAvailability,
  generateAvailabilityFromFile,
  checkAvailability,
  getAvailableSlots
};

// Example usage when run directly
if (require.main === module) {
  // Example 1: Generate from restaurant IDs
  console.log('Example 1: Generating availability for specific restaurant IDs...\n');
  const restaurantIds = ['r1', 'r2', 'r3', 'r4', 'r5'];
  const result1 = generateAvailability(restaurantIds, 7, false);
  console.log('Generated availability (first 500 chars):');
  console.log(result1.substring(0, 500) + '...\n');
  
  // Example 2: Generate from restaurants.json file
  console.log('Example 2: Generating availability from restaurants.json file...\n');
  try {
    const result2 = generateAvailabilityFromFile('restaurants.json', 14, true);
    console.log('Successfully generated and saved availability.json\n');
    
    // Example 3: Check specific availability
    const availabilityData = JSON.parse(result2);
    const available = checkAvailability(availabilityData, 'r1', '2024-01-20', '19:00');
    console.log(`Example 3: Tables available at r1 on 2024-01-20 at 19:00: ${available}\n`);
    
    // Example 4: Get all available slots
    const slots = getAvailableSlots(availabilityData, 'r1', '2024-01-20', 1);
    console.log('Example 4: Available slots for r1 on 2024-01-20:');
    console.log(JSON.stringify(slots.slice(0, 5), null, 2));
  } catch (error) {
    console.log('Could not read restaurants.json - make sure to generate it first using restaurantGenerator.js');
  }
}
