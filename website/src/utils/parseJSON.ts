/**
 * Leaderboard data loading utilities
 */

import type { LeaderboardEntry } from '../types/leaderboard';

/**
 * Raw JSON entry from leaderboard.json
 */
interface RawLeaderboardEntry {
  rank: number;
  model: string;
  provider: string;
  repo: string;
  mean: number | null;
  teleqna: number | null;
  teleqna_stderr: number | null;
  telelogs: number | null;
  telelogs_stderr: number | null;
  telemath: number | null;
  telemath_stderr: number | null;
  tsg: number | null;
  tsg_stderr: number | null;
}

/**
 * Parse JSON data into LeaderboardEntry array
 */
export function parseJSON(data: RawLeaderboardEntry[]): LeaderboardEntry[] {
  return data.map(entry => ({
    rank: entry.rank,
    provider: entry.provider,
    model: entry.model,
    repo: entry.repo,
    mean: entry.mean,
    teleqna: entry.teleqna,
    teleqna_stderr: entry.teleqna_stderr,
    telelogs: entry.telelogs,
    telelogs_stderr: entry.telelogs_stderr,
    telemath: entry.telemath,
    telemath_stderr: entry.telemath_stderr,
    tsg: entry.tsg,
    tsg_stderr: entry.tsg_stderr,
  }));
}

/**
 * Fetch and parse leaderboard data from JSON file
 */
export async function fetchLeaderboardData(): Promise<LeaderboardEntry[]> {
  const response = await fetch('/open_telco/data/leaderboard.json');
  if (!response.ok) {
    throw new Error('Failed to load leaderboard data');
  }
  const jsonData = await response.json();
  return parseJSON(jsonData);
}

