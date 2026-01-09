/**
 * Benchmark configurations and metadata
 */

import type { BenchmarkConfig } from '../types/leaderboard';

/**
 * Full benchmark configurations with metadata
 */
export const BENCHMARKS: BenchmarkConfig[] = [
  {
    key: 'teleqna',
    title: 'TeleQnA',
    description: '10,000 Q&A pairs testing telecom knowledge across lexicon, research, and standards.',
    icon: '📚',
    samples: '10,000',
    difficulty: 'medium',
    paperLink: 'https://arxiv.org/abs/2310.15051',
    datasetLink: 'https://huggingface.co/datasets/netop/TeleQnA',
  },
  {
    key: 'telemath',
    title: 'TeleMath',
    description: 'Mathematical reasoning in telecommunications: signal processing, network optimization.',
    icon: '🧮',
    samples: '500',
    difficulty: 'hard',
    paperLink: 'https://arxiv.org/abs/2506.10674',
    datasetLink: 'https://huggingface.co/datasets/netop/TeleMath',
  },
  {
    key: 'telelogs',
    title: 'TeleLogs',
    description: 'Root cause analysis for 5G network throughput degradation across 8 failure modes.',
    icon: '🔍',
    samples: '1,000+',
    difficulty: 'hard',
    paperLink: 'https://arxiv.org/abs/2507.21974',
    datasetLink: 'https://huggingface.co/datasets/netop/TeleLogs',
  },
  {
    key: 'tsg',
    title: '3GPP-TSG',
    description: 'Classification of technical documents by 3GPP working group (RAN, SA, CT).',
    icon: '📋',
    samples: '5,000+',
    difficulty: 'medium',
    datasetLink: 'https://huggingface.co/datasets/eaguaida/gsma_sample',
  },
];

/**
 * Leaderboard page benchmark cards (simpler version)
 */
export const LEADERBOARD_BENCHMARKS = BENCHMARKS.map(b => ({
  key: b.key,
  title: b.title,
  description: b.description,
  icon: b.icon,
}));

/**
 * TCI calculation parameters
 */
export const TCI_CONFIG = {
  /**
   * Benchmark difficulties (estimated based on average scores - lower avg = harder)
   * Higher value = easier benchmark
   */
  benchmarkDifficulty: {
    teleqna: 0.7,   // Easier - higher avg scores
    telelogs: 0.3,  // Harder - lower avg scores
    telemath: 0.4,  // Medium-hard
    tsg: 0.4,       // Medium-hard
  } as Record<string, number>,

  /**
   * Benchmark slopes (how discriminating each benchmark is)
   * Higher value = more discriminating
   */
  benchmarkSlope: {
    teleqna: 1.2,
    telelogs: 1.5,
    telemath: 1.3,
    tsg: 1.2,
  } as Record<string, number>,

  /**
   * Base error values for error bar calculations
   */
  baseErrors: {
    teleqna: 1.5,
    telelogs: 3.6,
    telemath: 2.8,
    tsg: 2.4,
    tci: 1.8,
  } as Record<string, number>,

  /**
   * Minimum number of benchmark scores required to calculate TCI
   */
  minScoresRequired: 3,

  /**
   * TCI scale parameters
   */
  baseScore: 115,
  scaleFactor: 20,
};

/**
 * Get difficulty badge color
 */
export function getDifficultyColor(difficulty: 'easy' | 'medium' | 'hard'): string {
  switch (difficulty) {
    case 'easy': return '#22c55e';
    case 'medium': return '#f59e0b';
    case 'hard': return '#ef4444';
  }
}

/**
 * Get TCI color based on score tier
 */
export function getTCIColor(tci: number | null): string {
  if (tci === null) return '#999';
  if (tci >= 135) return '#4DB6AC'; // High - teal
  if (tci >= 125) return '#81C784'; // Good - green
  if (tci >= 115) return '#FFB74D'; // Medium - amber
  return '#FFAB91'; // Lower - coral
}
