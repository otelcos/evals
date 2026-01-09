import React from 'react';

interface BenchmarkCardProps {
  title: string;
  description: string;
  icon?: React.ReactNode;
  sampleCount?: string;
  difficulty?: 'easy' | 'medium' | 'hard';
  link?: string;
  paperLink?: string;
  datasetLink?: string;
}

const getDifficultyClass = (difficulty: 'easy' | 'medium' | 'hard') => {
  switch (difficulty) {
    case 'easy': return 'badge-easy';
    case 'medium': return 'badge-medium';
    case 'hard': return 'badge-hard';
  }
};

export default function BenchmarkCard({
  title,
  description,
  icon,
  sampleCount,
  difficulty,
  link,
  paperLink,
  datasetLink,
}: BenchmarkCardProps): JSX.Element {
  return (
    <div className="benchmark-card-enhanced">
      <div className="benchmark-card-header">
        {icon && (
          <div className="benchmark-card-icon">
            {icon}
          </div>
        )}
        <div className="benchmark-card-title-area">
          <h3 className="benchmark-card-title">{title}</h3>
          <div className="benchmark-card-badges">
            {sampleCount && (
              <span className="badge badge-samples">{sampleCount}</span>
            )}
            {difficulty && (
              <span className={`badge ${getDifficultyClass(difficulty)}`}>
                {difficulty}
              </span>
            )}
          </div>
        </div>
        {(paperLink || datasetLink) && (
          <div style={{ display: 'flex', gap: '8px' }}>
            {paperLink && (
              <a
                href={paperLink}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  color: '#008080',
                  fontSize: '14px',
                }}
                title="Paper"
              >
                ↗
              </a>
            )}
          </div>
        )}
      </div>
      <p className="benchmark-card-description">{description}</p>
      {(paperLink || datasetLink) && (
        <div className="benchmark-card-footer">
          {paperLink && (
            <a
              href={paperLink}
              target="_blank"
              rel="noopener noreferrer"
            >
              Paper
            </a>
          )}
          {datasetLink && (
            <a
              href={datasetLink}
              target="_blank"
              rel="noopener noreferrer"
            >
              Dataset
            </a>
          )}
        </div>
      )}
    </div>
  );
}
