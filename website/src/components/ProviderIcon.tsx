import React from 'react';
import { getProviderLogoUrl, getProviderColor } from '../constants/providers';

interface ProviderIconProps {
  provider: string;
  size?: number;
  className?: string;
  showFallback?: boolean;
}

export default function ProviderIcon({
  provider,
  size = 24,
  className = '',
  showFallback = true,
}: ProviderIconProps): JSX.Element | null {
  const logoUrl = getProviderLogoUrl(provider);
  const color = getProviderColor(provider);

  if (logoUrl) {
    return (
      <img
        src={logoUrl}
        alt={provider}
        className={className}
        style={{
          width: size,
          height: size,
          borderRadius: '50%',
          objectFit: 'cover',
        }}
        onError={(e) => {
          (e.target as HTMLImageElement).style.display = 'none';
        }}
      />
    );
  }

  if (showFallback) {
    return (
      <div
        className={className}
        style={{
          width: size,
          height: size,
          borderRadius: '50%',
          backgroundColor: color,
          opacity: 0.8,
        }}
      />
    );
  }

  return null;
}
