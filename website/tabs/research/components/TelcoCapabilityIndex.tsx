import React, { useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Label,
  LabelList,
} from 'recharts';
import type { TCIDataPoint } from '../../../src/types/leaderboard';
import { useLeaderboardData } from '../../../src/hooks/useLeaderboardData';
import { calculateTCI } from '../../../src/utils/calculateTCI';
import { getProviderColor } from '../../../src/constants/providers';
import ProviderIcon from '../../../src/components/ProviderIcon';

const labeledModels = new Set([
  'gpt-5.2',
  'claude-opus-4.5',
  'gemini-3-flash-preview',
  'deepseek-v3.2',
]);

const labelOffsets: Record<string, { x: number; y: number }> = {
  'gpt-5.2': { x: -60, y: 0 },
  'claude-opus-4.5': { x: -110, y: 0 },
  'gemini-3-flash-preview': { x: -145, y: 0 },
  'deepseek-v3.2': { x: -95, y: 0 },
};

const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: TCIDataPoint }> }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div style={{
        backgroundColor: 'white',
        padding: '12px 16px',
        border: '1px solid #e0e0e0',
        borderRadius: '8px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
        fontSize: '13px',
        lineHeight: '1.5',
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '8px', color: data.color }}>
          {data.model}
        </div>
        <div style={{ color: '#666', marginBottom: '6px' }}>{data.provider}</div>
        <div style={{ fontWeight: 'bold', fontSize: '16px', marginBottom: '8px' }}>
          TCI Score: {data.tci}
        </div>
        <div style={{ borderTop: '1px solid #eee', paddingTop: '8px', fontSize: '12px' }}>
          <div>TeleQnA: {data.teleqna !== null ? `${data.teleqna}%` : '—'}</div>
          <div>TeleLogs: {data.telelogs !== null ? `${data.telelogs}%` : '—'}</div>
          <div>TeleMath: {data.telemath !== null ? `${data.telemath}%` : '—'}</div>
          <div>3GPP-TSG: {data.tsg !== null ? `${data.tsg}%` : '—'}</div>
        </div>
      </div>
    );
  }
  return null;
};

const renderCustomLabel = (props: { x?: number; y?: number; payload?: TCIDataPoint }) => {
  const { x, y, payload } = props;
  if (!x || !y || !payload?.isLabeled) return null;

  const offset = labelOffsets[payload.model] || { x: -80, y: 0 };

  return (
    <text
      x={x + offset.x}
      y={y + offset.y}
      fill={payload.color}
      fontSize={10}
      fontWeight="500"
      dominantBaseline="middle"
    >
      {payload.model}
    </text>
  );
};

export default function TelcoCapabilityIndex(): JSX.Element {
  const { data: leaderboardData, loading, error } = useLeaderboardData();

  const chartData = useMemo((): TCIDataPoint[] => {
    return leaderboardData
      .map((entry) => {
        const tci = calculateTCI(entry);
        if (tci === null) return null;

        return {
          rank: entry.rank,
          tci,
          model: entry.model,
          provider: entry.provider,
          color: getProviderColor(entry.provider),
          isLabeled: labeledModels.has(entry.model),
          teleqna: entry.teleqna,
          telelogs: entry.telelogs,
          telemath: entry.telemath,
          tsg: entry.tsg,
        };
      })
      .filter((d): d is TCIDataPoint => d !== null);
  }, [leaderboardData]);

  const providers = useMemo(() => {
    const seen = new Set<string>();
    const result: { name: string; color: string }[] = [];
    leaderboardData.forEach(entry => {
      if (!seen.has(entry.provider)) {
        seen.add(entry.provider);
        result.push({
          name: entry.provider,
          color: getProviderColor(entry.provider),
        });
      }
    });
    return result;
  }, [leaderboardData]);

  const yAxisDomain = useMemo(() => {
    if (chartData.length === 0) return [90, 140];
    const tciValues = chartData.map(d => d.tci);
    const minTCI = Math.min(...tciValues);
    const maxTCI = Math.max(...tciValues);
    const padding = (maxTCI - minTCI) * 0.1;
    return [
      Math.floor((minTCI - padding) / 5) * 5,
      Math.ceil((maxTCI + padding) / 5) * 5
    ];
  }, [chartData]);

  const yAxisTicks = useMemo(() => {
    const [min, max] = yAxisDomain;
    const ticks: number[] = [];
    for (let i = min; i <= max; i += 10) {
      ticks.push(i);
    }
    return ticks;
  }, [yAxisDomain]);

  if (loading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: '#666' }}>
        Loading leaderboard data...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: '#E60000' }}>
        Error loading data: {error}
      </div>
    );
  }

  return (
    <div style={{ width: '100%', padding: '20px 0' }}>
      <h2 style={{
        fontSize: '20px',
        fontWeight: 'bold',
        marginBottom: '8px',
        color: '#008080',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      }}>
        Telco Capabilities Index (TCI)
      </h2>
      <p style={{
        fontSize: '14px',
        color: '#6b7280',
        marginBottom: '20px',
        lineHeight: '1.5',
      }}>
        A unified measure of AI model performance across telecommunications-specific tasks,
        using IRT-inspired methodology for meaningful cross-model comparisons.
      </p>

      <div style={{ position: 'relative' }}>
        {/* Legend */}
        <div style={{
          position: 'absolute',
          top: '10px',
          right: '20px',
          backgroundColor: 'rgba(255,255,255,0.97)',
          border: '1px solid #E0E0E0',
          borderRadius: '6px',
          padding: '10px 14px',
          zIndex: 10,
          boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
        }}>
          <div style={{ fontSize: '9px', color: '#888', marginBottom: '6px' }}>
            {chartData.length} Results
          </div>
          <div style={{ fontSize: '10px', fontWeight: '600', marginBottom: '8px', color: '#333' }}>
            Organization
          </div>
          {providers.map((p) => (
            <div key={p.name} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
              <ProviderIcon provider={p.name} size={14} />
              <span style={{ fontSize: '10px', color: '#555' }}>{p.name}</span>
            </div>
          ))}
        </div>

        <ResponsiveContainer width="100%" height={480}>
          <ScatterChart margin={{ top: 20, right: 180, bottom: 50, left: 50 }}>
            <CartesianGrid
              strokeDasharray="0"
              stroke="#CCCCCC"
              strokeOpacity={0.25}
              vertical={true}
              horizontal={true}
            />
            <XAxis
              type="number"
              dataKey="rank"
              name="Rank"
              domain={[23, 0]}
              reversed={true}
              ticks={[20, 15, 10, 5, 1]}
              tick={{ fontSize: 11, fill: '#666' }}
              axisLine={false}
              tickLine={false}
            >
              <Label
                value="Leaderboard rank"
                offset={-5}
                position="bottom"
                style={{ fontSize: '12px', fill: '#666' }}
              />
            </XAxis>
            <YAxis
              type="number"
              dataKey="tci"
              name="Score"
              domain={yAxisDomain}
              ticks={yAxisTicks}
              tick={{ fontSize: 11, fill: '#666' }}
              axisLine={false}
              tickLine={false}
            >
              <Label
                value="Score"
                angle={0}
                position="top"
                offset={10}
                style={{ fontSize: '12px', fill: '#666' }}
              />
            </YAxis>
            <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
            <Scatter data={chartData} fill="#8884d8">
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.color}
                  fillOpacity={0.9}
                  r={9}
                />
              ))}
              <LabelList dataKey="model" content={renderCustomLabel} />
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
