import React from 'react';
import type { RankingEntry } from '../types/leaderboard';
import RankingRow from './RankingRow';

export type { RankingEntry };

interface LeaderboardCardProps {
  title: string;
  description: string;
  icon?: string;
  rankings: RankingEntry[];
  maxScore?: number;
  onViewFullRanking?: () => void;
}

export default function LeaderboardCard({
  title,
  description,
  icon,
  rankings,
  maxScore = 100,
  onViewFullRanking,
}: LeaderboardCardProps): JSX.Element {
  const topScore = rankings.length > 0 ? Math.max(...rankings.map(r => r.score)) : maxScore;
  const getBarWidth = (score: number) => Math.max(5, (score / topScore) * 100);

  return (
    <div className="leaderboard-card">
      <div className="leaderboard-card-header">
        {icon && <span className="leaderboard-card-icon">{icon}</span>}
        <div className="leaderboard-card-title-section">
          <h3 className="leaderboard-card-title">{title}</h3>
          <p className="leaderboard-card-description">{description}</p>
        </div>
      </div>

      <div className="leaderboard-rankings">
        {rankings.slice(0, 5).map((entry, index) => (
          <RankingRow
            key={`${entry.model}-${index}`}
            rank={entry.rank}
            model={entry.model}
            provider={entry.provider}
            score={entry.score}
            error={entry.error}
            barWidth={getBarWidth(entry.score)}
            isNew={entry.isNew}
          />
        ))}
      </div>

      {onViewFullRanking && (
        <button className="view-full-ranking" onClick={onViewFullRanking}>
          View Full Ranking <span className="arrow">&rarr;</span>
        </button>
      )}
    </div>
  );
}
