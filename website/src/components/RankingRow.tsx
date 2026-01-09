import React from 'react';
import { getProviderColor } from '../constants/providers';
import ProviderIcon from './ProviderIcon';

interface RankingRowProps {
  rank: number;
  model: string;
  provider: string;
  score: number;
  error: number;
  barWidth: number;
  showProviderIcon?: boolean;
  isNew?: boolean;
}

export default function RankingRow({
  rank,
  model,
  provider,
  score,
  error,
  barWidth,
  showProviderIcon = false,
  isNew = false,
}: RankingRowProps): JSX.Element {
  const barColor = getProviderColor(provider);

  return (
    <div className="leaderboard-ranking-row">
      <div className="ranking-position">{rank}</div>
      <div className="ranking-model-info">
        {showProviderIcon && (
          <ProviderIcon provider={provider} size={20} className="provider-logo" />
        )}
        <span className="ranking-model-name">
          {model}
          {isNew && <span className="new-badge">NEW</span>}
        </span>
      </div>
      <div className="ranking-bar-container">
        <div
          className="ranking-bar"
          style={{
            width: `${barWidth}%`,
            backgroundColor: barColor,
          }}
        >
          <div
            className="error-bar"
            style={{
              '--error-width': `${Math.max(3, error * 2)}%`,
            } as React.CSSProperties}
          >
            <div className="error-bar-whisker" />
          </div>
        </div>
      </div>
      <div className="ranking-score">
        <span className="score-value">{score.toFixed(2)}</span>
        <span className="score-error">{'\u00B1'}{error.toFixed(2)}</span>
      </div>
    </div>
  );
}
