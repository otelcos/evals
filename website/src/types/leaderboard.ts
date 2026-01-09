/**
 * Shared type definitions for leaderboard data structures
 */

/**
 * Raw entry from the leaderboard Parquet file
 */
export interface LeaderboardEntry {
  rank: number;
  provider: string;
  model: string;
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
  tci?: number | null;
}

/**
 * Entry for benchmark-specific rankings (used by LeaderboardCard)
 */
export interface RankingEntry {
  rank: number;
  model: string;
  provider: string;
  score: number;
  error: number;
  isNew?: boolean;
}

/**
 * Entry for TCI rankings (used by TCIHeroCard)
 */
export interface TCIEntry {
  rank: number;
  model: string;
  provider: string;
  tci: number;
  error: number;
  isNew?: boolean;
}

/**
 * Data point for scatter chart visualization
 */
export interface TCIDataPoint {
  rank: number;
  tci: number;
  model: string;
  provider: string;
  color: string;
  isLabeled: boolean;
  teleqna: number | null;
  telelogs: number | null;
  telemath: number | null;
  tsg: number | null;
}

/**
 * Benchmark configuration
 */
export interface BenchmarkConfig {
  key: string;
  title: string;
  description: string;
  icon?: string;
  samples?: string;
  difficulty?: 'easy' | 'medium' | 'hard';
  paperLink?: string;
  datasetLink?: string;
}
