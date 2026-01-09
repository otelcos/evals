import React, { useMemo, useState } from 'react';
import { useLeaderboardData } from '../../../src/hooks/useLeaderboardData';
import { withTCI } from '../../../src/utils/calculateTCI';
import { getTCIColor } from '../../../src/constants/benchmarks';
import ProviderIcon from '../../../src/components/ProviderIcon';

export default function ModelsPage(): JSX.Element {
  const { data: rawData, loading, error } = useLeaderboardData();
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'tci' | 'mean' | 'name'>('tci');
  const [selectedProviders, setSelectedProviders] = useState<Set<string>>(new Set());

  const data = useMemo(() => rawData.map(withTCI), [rawData]);

  const providers = useMemo(() => {
    return [...new Set(data.map(d => d.provider))].sort();
  }, [data]);

  const filteredData = useMemo(() => {
    let result = [...data];

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(d =>
        d.model.toLowerCase().includes(query) ||
        d.provider.toLowerCase().includes(query)
      );
    }

    if (selectedProviders.size > 0) {
      result = result.filter(d => selectedProviders.has(d.provider));
    }

    result.sort((a, b) => {
      if (sortBy === 'tci') {
        if (a.tci === null && b.tci === null) return 0;
        if (a.tci === null) return 1;
        if (b.tci === null) return -1;
        return b.tci - a.tci;
      }
      if (sortBy === 'mean') {
        if (a.mean === null && b.mean === null) return 0;
        if (a.mean === null) return 1;
        if (b.mean === null) return -1;
        return b.mean - a.mean;
      }
      return a.model.localeCompare(b.model);
    });

    return result;
  }, [data, searchQuery, selectedProviders, sortBy]);

  const toggleProvider = (provider: string) => {
    setSelectedProviders(prev => {
      const next = new Set(prev);
      if (next.has(provider)) {
        next.delete(provider);
      } else {
        next.add(provider);
      }
      return next;
    });
  };

  if (loading) {
    return <div style={{ padding: '40px', textAlign: 'center', color: '#666' }}>Loading models...</div>;
  }

  if (error) {
    return <div style={{ padding: '40px', textAlign: 'center', color: '#E60000' }}>Error: {error}</div>;
  }

  return (
    <div style={{ display: 'flex', gap: '40px', padding: '20px 0' }}>
      {/* Main content */}
      <div style={{ flex: 1 }}>
        <div style={{ marginBottom: '24px', color: '#666', fontSize: '15px' }}>
          {filteredData.length} models
        </div>

        {/* Model list */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
          {filteredData.map((model) => (
            <div
              key={`${model.provider}-${model.model}`}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '16px 0',
                borderBottom: '1px solid #eee',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <ProviderIcon provider={model.provider} size={24} />
                <div>
                  <div style={{ fontWeight: '600', fontSize: '15px', color: '#1a1a1a' }}>
                    {model.model}
                  </div>
                  <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
                    <span style={{
                      fontSize: '12px',
                      padding: '2px 8px',
                      backgroundColor: '#f5f5f5',
                      borderRadius: '4px',
                      color: '#555',
                    }}>
                      {model.provider}
                    </span>
                  </div>
                </div>
              </div>

              {/* TCI Score */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '50%',
                  backgroundColor: getTCIColor(model.tci),
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontWeight: '600',
                  fontSize: '14px',
                }}>
                  {model.tci !== null ? Math.round(model.tci) : '—'}
                </div>
                <span style={{ fontSize: '12px', color: '#888' }}>TCI</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Sidebar - Search & Filter */}
      <div style={{ width: '280px', flexShrink: 0 }}>
        <h3 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '20px', color: '#1a1a1a' }}>
          Search & Filter
        </h3>

        {/* Search input */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            border: '1px solid #e0e0e0',
            borderRadius: '8px',
            padding: '8px 12px',
          }}>
            <input
              type="text"
              placeholder="Search models"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                flex: 1,
                border: 'none',
                outline: 'none',
                fontSize: '14px',
                backgroundColor: 'transparent',
              }}
            />
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#999" strokeWidth="2">
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.35-4.35" />
            </svg>
          </div>
        </div>

        {/* Sort by */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{ fontSize: '11px', fontWeight: '600', color: '#888', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Sort by:
          </div>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as 'tci' | 'mean' | 'name')}
            style={{
              width: '100%',
              padding: '10px 12px',
              border: '1px solid #e0e0e0',
              borderRadius: '8px',
              fontSize: '14px',
              backgroundColor: 'white',
              cursor: 'pointer',
            }}
          >
            <option value="tci">Telco Capability Index</option>
            <option value="mean">Mean Score</option>
            <option value="name">Name</option>
          </select>
        </div>

        {/* Developer filter */}
        <div>
          <div style={{ fontSize: '11px', fontWeight: '600', color: '#888', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Developer
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '300px', overflowY: 'auto' }}>
            {providers.map(provider => (
              <label
                key={provider}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  cursor: 'pointer',
                  fontSize: '14px',
                  color: '#333',
                }}
              >
                <input
                  type="checkbox"
                  checked={selectedProviders.has(provider)}
                  onChange={() => toggleProvider(provider)}
                  style={{
                    width: '16px',
                    height: '16px',
                    cursor: 'pointer',
                  }}
                />
                {provider}
              </label>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
