/**
 * Category color mapping for blog post tags
 * Cohere-inspired with GSMA brand colors
 */
export const categoryColors: Record<string, { text: string; border: string }> = {
  Research: { text: '#9333ea', border: '#9333ea' },      // Purple
  News: { text: '#DC002B', border: '#DC002B' },          // GSMA Red
  Tutorial: { text: '#2563eb', border: '#2563eb' },      // Blue
  Product: { text: '#2563eb', border: '#2563eb' },       // Blue
  Newsroom: { text: '#DC002B', border: '#DC002B' },      // Red
  Company: { text: '#16a34a', border: '#16a34a' },       // Green
  'For Business': { text: '#0891b2', border: '#0891b2' }, // Cyan
  Developers: { text: '#7c3aed', border: '#7c3aed' },    // Violet
};

/**
 * Get colors for a category, with fallback for unknown categories
 */
export function getCategoryColors(category: string): { text: string; border: string } {
  return categoryColors[category] || { text: '#6b7280', border: '#d1d5db' };
}
