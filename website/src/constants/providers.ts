/**
 * Provider colors and logos for consistent branding across components
 */

/**
 * Color palette for each provider (used in charts and visualizations)
 */
export const PROVIDER_COLORS: Record<string, string> = {
  'Google': '#4DB6AC',
  'OpenAI': '#F48FB1',
  'Meta': '#FFAB91',
  'Anthropic': '#B39DDB',
  'Claude': '#B39DDB',
  'Grok': '#5C6BC0',
  'Qwen': '#81C784',
  'Mistral': '#FF8A65',
  'NetoAI': '#4DD0E1',
  'IBM': '#64B5F6',
  'IBM Granite': '#64B5F6',
  'DeepSeek': '#CE93D8',
  'LiquidAI': '#FFB74D',
  'Microsoft': '#4FC3F7',
  'Swiss AI': '#E57373',
  'ByteDance': '#AED581',
  'Amazon': '#FF9800',
  'NVIDIA': '#76B900',
  'Cohere': '#FF6B6B',
  'Hugging Face': '#FFD54F',
  'Other': '#A1887F',
};

/**
 * Logo filenames for each provider (stored in /static/img/logos/)
 */
export const PROVIDER_LOGOS: Record<string, string> = {
  'Google': 'deepmind.png',
  'OpenAI': 'openai.png',
  'Meta': 'meta.png',
  'Anthropic': 'anthropic.png',
  'Claude': 'anthropic.png',
  'Grok': 'xai.png',
  'Qwen': 'qwen.png',
  'Mistral': 'mistral.png',
  'IBM': 'ibm.png',
  'IBM Granite': 'ibm.png',
  'DeepSeek': 'deepseek.png',
  'LiquidAI': 'liquidai.png',
  'Microsoft': 'microsoft.png',
  'ByteDance': 'bytedance.png',
  'Amazon': 'amazon.png',
  'NVIDIA': 'nvidia.png',
  'Cohere': 'cohere.png',
  'Hugging Face': 'huggingface.png',
  'NetoAI': 'NetoAI-logo.png',
};

/**
 * Get provider color with fallback
 */
export function getProviderColor(provider: string): string {
  return PROVIDER_COLORS[provider] || PROVIDER_COLORS['Other'];
}

/**
 * Get provider logo filename (returns undefined if not found)
 */
export function getProviderLogo(provider: string): string | undefined {
  return PROVIDER_LOGOS[provider];
}

/**
 * Base path for logo images
 */
export const LOGO_BASE_PATH = '/open_telco/img/logos/';

/**
 * Get full logo URL for a provider
 */
export function getProviderLogoUrl(provider: string): string | undefined {
  const logo = PROVIDER_LOGOS[provider];
  return logo ? `${LOGO_BASE_PATH}${logo}` : undefined;
}
