import { useState, useEffect } from 'react';
import type { LeaderboardEntry } from '../types/leaderboard';
import { parseJSON } from '../utils/parseJSON';

interface UseLeaderboardDataResult {
  data: LeaderboardEntry[];
  loading: boolean;
  error: string | null;
}

export function useLeaderboardData(): UseLeaderboardDataResult {
  const [data, setData] = useState<LeaderboardEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/open_telco/data/leaderboard.json')
      .then(response => {
        if (!response.ok) throw new Error('Failed to load leaderboard data');
        return response.json();
      })
      .then(jsonData => {
        setData(parseJSON(jsonData));
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return { data, loading, error };
}
