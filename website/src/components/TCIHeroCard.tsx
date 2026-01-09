import React from 'react';
import type { TCIEntry } from '../types/leaderboard';
import RankingRow from './RankingRow';

export type { TCIEntry };

interface TCIHeroCardProps {
  rankings: TCIEntry[];
  onViewFullRanking?: () => void;
  expanded?: boolean;
}

export default function TCIHeroCard({
  rankings,
  onViewFullRanking,
  expanded = false,
}: TCIHeroCardProps): JSX.Element {
  const minTCI = 90;
  const maxTCI = Math.max(...rankings.map(r => r.tci), 150);
  const range = maxTCI - minTCI;
  const getBarWidth = (tci: number) => Math.max(5, ((tci - minTCI) / range) * 100);

  const displayRankings = expanded ? rankings : rankings.slice(0, 5);

  return (
    <div className={`leaderboard-card tci-hero-card ${expanded ? 'expanded' : ''}`}>
      <div className="leaderboard-card-header">
        <span className="leaderboard-card-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 3V21H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M7 14L11 10L15 14L21 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </span>
        <div className="leaderboard-card-title-section">
          <h3 className="leaderboard-card-title">Telco Capability Index (TCI)</h3>
          <p className="leaderboard-card-description">
            A unified measure of AI model performance across telecommunications-specific tasks,
            using IRT-inspired methodology for meaningful cross-model comparisons.
          </p>
        </div>
      </div>

      <div className="leaderboard-rankings">
        {displayRankings.map((entry, index) => (
          <RankingRow
            key={`${entry.model}-${index}`}
            rank={entry.rank}
            model={entry.model}
            provider={entry.provider}
            score={entry.tci}
            error={entry.error}
            barWidth={getBarWidth(entry.tci)}
            showProviderIcon={true}
            isNew={entry.isNew}
          />
        ))}
      </div>

      {onViewFullRanking && !expanded && (
        <button className="view-full-ranking" onClick={onViewFullRanking}>
          View Full Ranking <span className="arrow">&rarr;</span>
        </button>
      )}
    </div>
  );
}
