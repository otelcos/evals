/**
 * TCI (Telco Capability Index) calculation utilities
 *
 * Uses IRT-inspired methodology for meaningful cross-model comparisons
 */

import type { LeaderboardEntry } from '../types/leaderboard';
import { TCI_CONFIG } from '../constants/benchmarks';

/**
 * Calculate TCI score using IRT-inspired methodology
 *
 * The TCI score is calculated based on weighted performance across
 * four benchmarks (TeleQnA, TeleLogs, TeleMath, 3GPP-TSG), taking into
 * account each benchmark's difficulty and discrimination factor.
 *
 * @param entry - Leaderboard entry with benchmark scores
 * @returns TCI score (typically 90-150 range) or null if insufficient data
 */
export function calculateTCI(entry: LeaderboardEntry): number | null {
  const { teleqna, telelogs, telemath, tsg } = entry;

  // Need at least 3 scores to calculate TCI
  const scores = [teleqna, telelogs, telemath, tsg].filter(s => s !== null) as number[];
  if (scores.length < TCI_CONFIG.minScoresRequired) return null;

  const benchmarks = [
    { key: 'teleqna', value: teleqna },
    { key: 'telelogs', value: telelogs },
    { key: 'telemath', value: telemath },
    { key: 'tsg', value: tsg },
  ];

  let totalWeight = 0;
  let weightedCapability = 0;

  benchmarks.forEach(({ key, value }) => {
    if (value === null) return;

    // Normalize score to 0-1 range
    const score = value / 100;

    // Get difficulty and slope from config
    const difficulty = 1 - TCI_CONFIG.benchmarkDifficulty[key];
    const slope = TCI_CONFIG.benchmarkSlope[key];

    // Clamp score to prevent log(0) or log(inf)
    const adjustedScore = Math.max(0.01, Math.min(0.99, score));

    // Logit transformation (IRT-inspired)
    const logitScore = Math.log(adjustedScore / (1 - adjustedScore));

    // Weight by difficulty and slope
    const weight = difficulty * slope;
    weightedCapability += (logitScore + difficulty * 2) * weight;
    totalWeight += weight;
  });

  // Scale to TCI range (roughly 90-150)
  const rawCapability = weightedCapability / totalWeight;
  const tci = TCI_CONFIG.baseScore + rawCapability * TCI_CONFIG.scaleFactor;

  return Math.round(tci * 10) / 10;
}

/**
 * Calculate synthetic error based on score and benchmark difficulty
 *
 * Higher scores have lower error, lower scores have higher error.
 * Each benchmark has a different base error reflecting measurement uncertainty.
 *
 * @param score - The benchmark or TCI score
 * @param benchmarkKey - The benchmark identifier (e.g., 'teleqna', 'tci')
 * @returns Error margin value
 */
export function calculateError(score: number, benchmarkKey: string): number {
  const baseError = TCI_CONFIG.baseErrors[benchmarkKey] || 2.0;

  // Higher scores have lower error, lower scores have higher error
  return Math.round((baseError * (1 + (100 - score) / 200)) * 100) / 100;
}

/**
 * Calculate TCI and add it to a leaderboard entry
 *
 * @param entry - Leaderboard entry
 * @returns Entry with TCI calculated
 */
export function withTCI(entry: LeaderboardEntry): LeaderboardEntry {
  return {
    ...entry,
    tci: calculateTCI(entry),
  };
}

/**
 * Sort entries by TCI score (descending, nulls last)
 */
export function sortByTCI(entries: LeaderboardEntry[]): LeaderboardEntry[] {
  return [...entries].sort((a, b) => {
    const tciA = a.tci ?? calculateTCI(a);
    const tciB = b.tci ?? calculateTCI(b);

    if (tciA === null && tciB === null) return 0;
    if (tciA === null) return 1;
    if (tciB === null) return -1;
    return tciB - tciA;
  });
}
