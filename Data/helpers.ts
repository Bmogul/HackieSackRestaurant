import * as fs from 'fs';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface Restaurant {
  id: string;
  name: string;
  cuisine: string;
  location: string;
  price_range: string;
  rating: number;
  dietary_options: string[];
  hours: {
    lunch?: string;
    dinner: string;
  };
  phone: string;
  address: string;
  features: string[];
  popular_dishes: string[];
  average_price_per_person: string;
  reservations_required: boolean;
  outdoor_seating: boolean;
  private_dining: boolean;
  takeout_available: boolean;
  delivery_available: boolean;
}

export interface AvailabilityData {
  [restaurantId: string]: {
    [date: string]: {
      [time: string]: number;
    };
  };
}

export interface TimeSlot {
  time: string;
  available_tables: number;
}

export interface SearchFilters {
  cuisine?: string;
  location?: string;
  price_range?: string;
  dietary_options?: string[];
  min_rating?: number;
  outdoor_seating?: boolean;
  private_dining?: boolean;
}

export interface Booking {
  booking_id: string;
  restaurant_id: string;
  restaurant_name: string;
  date: string;
  time: string;
  party_size: number;
  customer_name: string;
  customer_phone: string;
  special_requests?: string;
  status: 'confirmed' | 'pending' | 'cancelled';
  created_at: string;
}

// ============================================================================
// FILE I/O HELPERS
// ============================================================================

/**
 * Load restaurants from JSON file
 */
export function loadRestaurants(filePath: string = 'restaurants.json'): Restaurant[] {
  try {
    const data = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    console.error(`Error loading restaurants from ${filePath}:`, error);
    return [];
  }
}

/**
 * Load availability from JSON file
 */
export function loadAvailability(filePath: string = 'availability.json'): AvailabilityData {
  try {
    const data = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    console.error(`Error loading availability from ${filePath}:`, error);
    return {};
  }
}

/**
 * Save data to JSON file
 */
export function saveToFile(data: any, filePath: string): boolean {
  try {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
    return true;
  } catch (error) {
    console.error(`Error saving to ${filePath}:`, error);
    return false;
  }
}

// ============================================================================
// RESTAURANT SEARCH & FILTER HELPERS
// ============================================================================

/**
 * Search restaurants by filters
 */
export function searchRestaurants(
  restaurants: Restaurant[],
  filters: SearchFilters
): Restaurant[] {
  return restaurants.filter(restaurant => {
    // Filter by cuisine
    if (filters.cuisine && restaurant.cuisine.toLowerCase() !== filters.cuisine.toLowerCase()) {
      return false;
    }

    // Filter by location
    if (filters.location && restaurant.location.toLowerCase() !== filters.location.toLowerCase()) {
      return false;
    }

    // Filter by price range
    if (filters.price_range && restaurant.price_range !== filters.price_range) {
      return false;
    }

    // Filter by dietary options
    if (filters.dietary_options && filters.dietary_options.length > 0) {
      const hasAllOptions = filters.dietary_options.every(option =>
        restaurant.dietary_options.some(ro => ro.toLowerCase() === option.toLowerCase())
      );
      if (!hasAllOptions) {
        return false;
      }
    }

    // Filter by minimum rating
    if (filters.min_rating && restaurant.rating < filters.min_rating) {
      return false;
    }

    // Filter by outdoor seating
    if (filters.outdoor_seating !== undefined && restaurant.outdoor_seating !== filters.outdoor_seating) {
      return false;
    }

    // Filter by private dining
    if (filters.private_dining !== undefined && restaurant.private_dining !== filters.private_dining) {
      return false;
    }

    return true;
  });
}

/**
 * Get restaurant by ID
 */
export function getRestaurantById(restaurants: Restaurant[], restaurantId: string): Restaurant | null {
  return restaurants.find(r => r.id === restaurantId) || null;
}

/**
 * Get restaurants by cuisine type
 */
export function getRestaurantsByCuisine(restaurants: Restaurant[], cuisine: string): Restaurant[] {
  return restaurants.filter(r => r.cuisine.toLowerCase() === cuisine.toLowerCase());
}

/**
 * Get restaurants by location
 */
export function getRestaurantsByLocation(restaurants: Restaurant[], location: string): Restaurant[] {
  return restaurants.filter(r => r.location.toLowerCase() === location.toLowerCase());
}

/**
 * Sort restaurants by rating (descending)
 */
export function sortByRating(restaurants: Restaurant[]): Restaurant[] {
  return [...restaurants].sort((a, b) => b.rating - a.rating);
}

/**
 * Sort restaurants by price (ascending)
 */
export function sortByPrice(restaurants: Restaurant[]): Restaurant[] {
  const priceOrder: { [key: string]: number } = { '$': 1, '$$': 2, '$$$': 3, '$$$$': 4 };
  return [...restaurants].sort((a, b) => priceOrder[a.price_range] - priceOrder[b.price_range]);
}

// ============================================================================
// AVAILABILITY HELPERS
// ============================================================================

/**
 * Check if a restaurant has availability at a specific date and time
 */
export function checkAvailability(
  availability: AvailabilityData,
  restaurantId: string,
  date: string,
  time: string
): number | null {
  return availability[restaurantId]?.[date]?.[time] ?? null;
}

/**
 * Get all available time slots for a restaurant on a specific date
 */
export function getAvailableSlots(
  availability: AvailabilityData,
  restaurantId: string,
  date: string,
  minTables: number = 1
): TimeSlot[] {
  const dateSlots = availability[restaurantId]?.[date];
  if (!dateSlots) return [];

  return Object.entries(dateSlots)
    .filter(([_, tables]) => tables >= minTables)
    .map(([time, tables]) => ({ time, available_tables: tables }))
    .sort((a, b) => a.time.localeCompare(b.time));
}

/**
 * Get available dates for a restaurant (dates with at least one available slot)
 */
export function getAvailableDates(
  availability: AvailabilityData,
  restaurantId: string,
  minTables: number = 1
): string[] {
  const restaurantAvailability = availability[restaurantId];
  if (!restaurantAvailability) return [];

  return Object.keys(restaurantAvailability)
    .filter(date => {
      const slots = restaurantAvailability[date];
      return Object.values(slots).some(tables => tables >= minTables);
    })
    .sort();
}

/**
 * Find alternative time slots if preferred time is not available
 */
export function findAlternativeSlots(
  availability: AvailabilityData,
  restaurantId: string,
  date: string,
  preferredTime: string,
  minTables: number = 1,
  maxAlternatives: number = 3
): TimeSlot[] {
  const allSlots = getAvailableSlots(availability, restaurantId, date, minTables);
  
  // Remove the preferred time if it exists
  const alternatives = allSlots.filter(slot => slot.time !== preferredTime);
  
  // Sort by proximity to preferred time
  const preferredMinutes = timeToMinutes(preferredTime);
  alternatives.sort((a, b) => {
    const diffA = Math.abs(timeToMinutes(a.time) - preferredMinutes);
    const diffB = Math.abs(timeToMinutes(b.time) - preferredMinutes);
    return diffA - diffB;
  });

  return alternatives.slice(0, maxAlternatives);
}

/**
 * Check availability across multiple restaurants
 */
export function checkMultipleRestaurants(
  availability: AvailabilityData,
  restaurantIds: string[],
  date: string,
  time: string,
  minTables: number = 1
): { restaurantId: string; available: boolean; tables: number }[] {
  return restaurantIds.map(restaurantId => {
    const tables = checkAvailability(availability, restaurantId, date, time) || 0;
    return {
      restaurantId,
      available: tables >= minTables,
      tables
    };
  });
}

// ============================================================================
// DATE & TIME HELPERS
// ============================================================================

/**
 * Convert time string (HH:MM) to minutes since midnight
 */
export function timeToMinutes(time: string): number {
  const [hours, minutes] = time.split(':').map(Number);
  return hours * 60 + minutes;
}

/**
 * Format date to YYYY-MM-DD
 */
export function formatDate(date: Date): string {
  return date.toISOString().split('T')[0];
}

/**
 * Get today's date as YYYY-MM-DD
 */
export function getToday(): string {
  return formatDate(new Date());
}

/**
 * Get date N days from today
 */
export function getDaysFromNow(days: number): string {
  const date = new Date();
  date.setDate(date.getDate() + days);
  return formatDate(date);
}

/**
 * Parse date string (YYYY-MM-DD) to Date object
 */
export function parseDate(dateString: string): Date {
  return new Date(dateString + 'T00:00:00');
}

/**
 * Check if date is in the past
 */
export function isDateInPast(dateString: string): boolean {
  const date = parseDate(dateString);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  return date < today;
}

/**
 * Get day of week name
 */
export function getDayOfWeek(dateString: string): string {
  const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const date = parseDate(dateString);
  return days[date.getDay()];
}

/**
 * Check if date is a weekend
 */
export function isWeekend(dateString: string): boolean {
  const date = parseDate(dateString);
  const day = date.getDay();
  return day === 0 || day === 6;
}

// ============================================================================
// BOOKING HELPERS
// ============================================================================

/**
 * Generate a unique booking ID
 */
export function generateBookingId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 7);
  return `BK${timestamp}${random}`.toUpperCase();
}

/**
 * Create a new booking
 */
export function createBooking(
  restaurantId: string,
  restaurantName: string,
  date: string,
  time: string,
  partySize: number,
  customerName: string,
  customerPhone: string,
  specialRequests?: string
): Booking {
  return {
    booking_id: generateBookingId(),
    restaurant_id: restaurantId,
    restaurant_name: restaurantName,
    date,
    time,
    party_size: partySize,
    customer_name: customerName,
    customer_phone: customerPhone,
    special_requests: specialRequests,
    status: 'confirmed',
    created_at: new Date().toISOString()
  };
}

/**
 * Validate booking details
 */
export function validateBooking(
  date: string,
  time: string,
  partySize: number
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Check if date is in the past
  if (isDateInPast(date)) {
    errors.push('Cannot book for a past date');
  }

  // Check party size
  if (partySize < 1) {
    errors.push('Party size must be at least 1');
  }
  if (partySize > 20) {
    errors.push('Party size cannot exceed 20. Please contact restaurant directly for large groups');
  }

  // Check time format
  if (!/^\d{2}:\d{2}$/.test(time)) {
    errors.push('Invalid time format. Use HH:MM');
  }

  return {
    valid: errors.length === 0,
    errors
  };
}

/**
 * Update availability after booking
 */
export function updateAvailabilityAfterBooking(
  availability: AvailabilityData,
  restaurantId: string,
  date: string,
  time: string,
  tablesBooked: number = 1
): AvailabilityData {
  const updated = JSON.parse(JSON.stringify(availability)); // Deep clone
  
  if (updated[restaurantId]?.[date]?.[time] !== undefined) {
    updated[restaurantId][date][time] = Math.max(0, updated[restaurantId][date][time] - tablesBooked);
  }
  
  return updated;
}

// ============================================================================
// RECOMMENDATION HELPERS
// ============================================================================

/**
 * Get restaurant recommendations based on preferences
 */
export function getRecommendations(
  restaurants: Restaurant[],
  preferences: {
    cuisine?: string;
    dietary_options?: string[];
    price_range?: string;
    min_rating?: number;
  },
  limit: number = 5
): Restaurant[] {
  let scored = restaurants.map(restaurant => {
    let score = 0;

    // Exact cuisine match
    if (preferences.cuisine && restaurant.cuisine.toLowerCase() === preferences.cuisine.toLowerCase()) {
      score += 10;
    }

    // Dietary options match
    if (preferences.dietary_options) {
      const matchCount = preferences.dietary_options.filter(option =>
        restaurant.dietary_options.some(ro => ro.toLowerCase() === option.toLowerCase())
      ).length;
      score += matchCount * 5;
    }

    // Price range match
    if (preferences.price_range && restaurant.price_range === preferences.price_range) {
      score += 5;
    }

    // Rating bonus
    score += restaurant.rating * 2;

    // Min rating filter
    if (preferences.min_rating && restaurant.rating < preferences.min_rating) {
      score = -1; // Exclude
    }

    return { restaurant, score };
  });

  return scored
    .filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(item => item.restaurant);
}

/**
 * Find similar restaurants
 */
export function findSimilarRestaurants(
  restaurants: Restaurant[],
  targetRestaurant: Restaurant,
  limit: number = 3
): Restaurant[] {
  const scored = restaurants
    .filter(r => r.id !== targetRestaurant.id)
    .map(restaurant => {
      let score = 0;

      // Same cuisine
      if (restaurant.cuisine === targetRestaurant.cuisine) {
        score += 10;
      }

      // Similar price range
      if (restaurant.price_range === targetRestaurant.price_range) {
        score += 5;
      }

      // Same location
      if (restaurant.location === targetRestaurant.location) {
        score += 3;
      }

      // Similar rating (within 0.5)
      if (Math.abs(restaurant.rating - targetRestaurant.rating) <= 0.5) {
        score += 2;
      }

      // Shared dietary options
      const sharedOptions = restaurant.dietary_options.filter(option =>
        targetRestaurant.dietary_options.includes(option)
      ).length;
      score += sharedOptions;

      return { restaurant, score };
    });

  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(item => item.restaurant);
}

// ============================================================================
// FORMATTING HELPERS
// ============================================================================

/**
 * Format restaurant for display
 */
export function formatRestaurantDisplay(restaurant: Restaurant): string {
  return `
${restaurant.name} (${restaurant.id})
Cuisine: ${restaurant.cuisine}
Location: ${restaurant.location}
Price: ${restaurant.price_range} (${restaurant.average_price_per_person})
Rating: ${restaurant.rating}/5.0
Dietary Options: ${restaurant.dietary_options.join(', ')}
Phone: ${restaurant.phone}
Address: ${restaurant.address}
Features: ${restaurant.features.join(', ')}
Popular Dishes: ${restaurant.popular_dishes.join(', ')}
  `.trim();
}

/**
 * Format booking confirmation
 */
export function formatBookingConfirmation(booking: Booking): string {
  return `
BOOKING CONFIRMATION
Confirmation Number: ${booking.booking_id}
Restaurant: ${booking.restaurant_name}
Date: ${booking.date} (${getDayOfWeek(booking.date)})
Time: ${booking.time}
Party Size: ${booking.party_size}
Name: ${booking.customer_name}
Phone: ${booking.customer_phone}
${booking.special_requests ? `Special Requests: ${booking.special_requests}` : ''}
Status: ${booking.status.toUpperCase()}
Booked on: ${new Date(booking.created_at).toLocaleString()}
  `.trim();
}

/**
 * Format price range to readable string
 */
export function formatPriceRange(priceSymbol: string): string {
  const ranges: { [key: string]: string } = {
    '$': 'Budget-friendly ($10-20)',
    '$$': 'Moderate ($20-40)',
    '$$$': 'Upscale ($40-70)',
    '$$$$': 'Fine Dining ($70-150)'
  };
  return ranges[priceSymbol] || priceSymbol;
}

// ============================================================================
// STATISTICS HELPERS
// ============================================================================

/**
 * Get restaurant statistics
 */
export function getRestaurantStats(restaurants: Restaurant[]): {
  total: number;
  by_cuisine: { [cuisine: string]: number };
  by_location: { [location: string]: number };
  by_price_range: { [price: string]: number };
  average_rating: number;
} {
  const stats = {
    total: restaurants.length,
    by_cuisine: {} as { [cuisine: string]: number },
    by_location: {} as { [location: string]: number },
    by_price_range: {} as { [price: string]: number },
    average_rating: 0
  };

  let totalRating = 0;

  restaurants.forEach(restaurant => {
    // Count by cuisine
    stats.by_cuisine[restaurant.cuisine] = (stats.by_cuisine[restaurant.cuisine] || 0) + 1;

    // Count by location
    stats.by_location[restaurant.location] = (stats.by_location[restaurant.location] || 0) + 1;

    // Count by price range
    stats.by_price_range[restaurant.price_range] = (stats.by_price_range[restaurant.price_range] || 0) + 1;

    // Sum ratings
    totalRating += restaurant.rating;
  });

  stats.average_rating = restaurants.length > 0 ? totalRating / restaurants.length : 0;

  return stats;
}

/**
 * Get availability statistics for a restaurant
 */
export function getAvailabilityStats(
  availability: AvailabilityData,
  restaurantId: string
): {
  total_slots: number;
  available_slots: number;
  fully_booked_slots: number;
  availability_rate: number;
} {
  const restaurantAvailability = availability[restaurantId];
  if (!restaurantAvailability) {
    return { total_slots: 0, available_slots: 0, fully_booked_slots: 0, availability_rate: 0 };
  }

  let totalSlots = 0;
  let availableSlots = 0;
  let fullyBookedSlots = 0;

  Object.values(restaurantAvailability).forEach(dateSlots => {
    Object.values(dateSlots).forEach(tables => {
      totalSlots++;
      if (tables > 0) {
        availableSlots++;
      } else {
        fullyBookedSlots++;
      }
    });
  });

  return {
    total_slots: totalSlots,
    available_slots: availableSlots,
    fully_booked_slots: fullyBookedSlots,
    availability_rate: totalSlots > 0 ? (availableSlots / totalSlots) * 100 : 0
  };
}

// ============================================================================
// EXPORT ALL
// ============================================================================

export default {
  // File I/O
  loadRestaurants,
  loadAvailability,
  saveToFile,
  
  // Search & Filter
  searchRestaurants,
  getRestaurantById,
  getRestaurantsByCuisine,
  getRestaurantsByLocation,
  sortByRating,
  sortByPrice,
  
  // Availability
  checkAvailability,
  getAvailableSlots,
  getAvailableDates,
  findAlternativeSlots,
  checkMultipleRestaurants,
  
  // Date & Time
  timeToMinutes,
  formatDate,
  getToday,
  getDaysFromNow,
  parseDate,
  isDateInPast,
  getDayOfWeek,
  isWeekend,
  
  // Booking
  generateBookingId,
  createBooking,
  validateBooking,
  updateAvailabilityAfterBooking,
  
  // Recommendations
  getRecommendations,
  findSimilarRestaurants,
  
  // Formatting
  formatRestaurantDisplay,
  formatBookingConfirmation,
  formatPriceRange,
  
  // Statistics
  getRestaurantStats,
  getAvailabilityStats
};
