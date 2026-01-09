import React, { useState, useMemo } from 'react';
import Layout from '@theme/Layout';
import LeaderboardCard from '../components/LeaderboardCard';
import TCIHeroCard from '../components/TCIHeroCard';
import type { LeaderboardEntry, RankingEntry, TCIEntry } from '../types/leaderboard';
import { useLeaderboardData } from '../hooks/useLeaderboardData';
import { calculateTCI, calculateError } from '../utils/calculateTCI';
import { LEADERBOARD_BENCHMARKS } from '../constants/benchmarks';

export default function LeaderboardPage(): JSX.Element {
  const { data, loading, error } = useLeaderboardData();
  const [expandedCard, setExpandedCard] = useState<string | null>(null);

  // Calculate TCI rankings
  const tciRankings = useMemo((): TCIEntry[] => {
    const rankings = data
      .map(entry => {
        const tci = calculateTCI(entry);
        if (tci === null) return null;
        return {
          rank: 0,
          model: entry.model,
          provider: entry.provider,
          tci,
          error: calculateError(tci, 'tci'),
          isNew: entry.rank <= 3, // Mark top 3 as "new" for demo
        };
      })
      .filter((e): e is TCIEntry => e !== null)
      .sort((a, b) => b.tci - a.tci)
      .map((entry, index) => ({ ...entry, rank: index + 1 }));

    return rankings;
  }, [data]);

  // Calculate benchmark rankings
  const getBenchmarkRankings = (benchmarkKey: string): RankingEntry[] => {
    const key = benchmarkKey as keyof LeaderboardEntry;
    return data
      .filter(entry => entry[key] !== null && typeof entry[key] === 'number')
      .map(entry => ({
        rank: 0,
        model: entry.model,
        provider: entry.provider,
        score: entry[key] as number,
        error: calculateError(entry[key] as number, benchmarkKey),
        isNew: entry.rank <= 3,
      }))
      .sort((a, b) => b.score - a.score)
      .map((entry, index) => ({ ...entry, rank: index + 1 }));
  };

  const handleViewFullRanking = (cardKey: string) => {
    setExpandedCard(expandedCard === cardKey ? null : cardKey);
  };

  if (loading) {
    return (
      <Layout title="Leaderboard" description="Open Telco LLM Leaderboard">
        <div className="leaderboard-page">
          <div className="leaderboard-loading">Loading leaderboard data...</div>
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout title="Leaderboard" description="Open Telco LLM Leaderboard">
        <div className="leaderboard-page">
          <div className="leaderboard-error">Error: {error}</div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title="Leaderboard" description="Open Telco LLM Leaderboard">
      <div className="leaderboard-page">
        <div className="leaderboard-header">
          <h1>Leaderboard</h1>
          <p>Track the performance of language models across Open Telco evaluations.</p>
        </div>

        {/* TCI Hero Card - Full Width */}
        <div className="tci-hero-wrapper">
          <TCIHeroCard
            rankings={tciRankings}
            expanded={expandedCard === 'tci'}
            onViewFullRanking={() => handleViewFullRanking('tci')}
          />
        </div>

        {/* Spacer */}
        <div className="leaderboard-section-divider" />

        {/* Section Header */}
        <div className="leaderboard-section-header">
          <h2>Individual Benchmarks</h2>
          <p>Performance breakdown across specific evaluation tasks</p>
        </div>

        {/* Benchmark Cards Grid - 3 columns */}
        <div className="leaderboard-benchmarks-grid">
          {LEADERBOARD_BENCHMARKS.map(benchmark => {
            const rankings = getBenchmarkRankings(benchmark.key);
            if (rankings.length === 0) return null;

            return (
              <LeaderboardCard
                key={benchmark.key}
                title={benchmark.title}
                description={benchmark.description}
                icon={benchmark.icon}
                rankings={expandedCard === benchmark.key ? rankings : rankings.slice(0, 5)}
                onViewFullRanking={() => handleViewFullRanking(benchmark.key)}
              />
            );
          })}
        </div>

        <div className="leaderboard-footer">
          <h3>Submission Guidelines</h3>
          <p>
            Want to add your model to the leaderboard? Run the full evaluation suite,
            document your setup, and submit via{' '}
            <a href="https://github.com/gsma-research/open_telco/issues" target="_blank" rel="noopener noreferrer">
              GitHub Issues
            </a>.
          </p>
        </div>
      </div>
    </Layout>
  );
}
