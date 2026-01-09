import React from 'react';

type InfoCardVariant = 'tip' | 'warning' | 'info' | 'success';

interface InfoCardProps {
  variant?: InfoCardVariant;
  title?: string;
  children: React.ReactNode;
}

const variantStyles: Record<InfoCardVariant, { bg: string; border: string; icon: string }> = {
  tip: { bg: '#f0fdf4', border: '#22c55e', icon: '💡' },
  warning: { bg: '#fffbeb', border: '#f59e0b', icon: '⚠️' },
  info: { bg: '#eff6ff', border: '#3b82f6', icon: 'ℹ️' },
  success: { bg: '#f0fdf4', border: '#22c55e', icon: '✓' },
};

export default function InfoCard({
  variant = 'info',
  title,
  children
}: InfoCardProps): JSX.Element {
  const style = variantStyles[variant];

  return (
    <div
      className="info-card"
      style={{
        background: style.bg,
        borderLeft: `4px solid ${style.border}`,
        borderRadius: '6px',
        padding: '16px 20px',
        margin: '16px 0',
      }}
    >
      {title && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginBottom: '8px',
          fontWeight: 600,
          fontSize: '15px',
        }}>
          <span>{style.icon}</span>
          <span>{title}</span>
        </div>
      )}
      <div style={{ fontSize: '14px', lineHeight: 1.6, color: '#374151' }}>
        {children}
      </div>
    </div>
  );
}
