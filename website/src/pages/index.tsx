import React, { useMemo } from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './index.module.css';
import { BENCHMARKS, getDifficultyColor } from '../constants/benchmarks';
import { useLeaderboardData } from '../hooks/useLeaderboardData';
import { calculateTCI } from '../utils/calculateTCI';

// Collaborators
const COLLABORATORS = {
  operators: [
    'AT&T', 'China Telecom', 'Deutsche Telekom', 'du', 'KDDI', 'KPN',
    'Liberty Global', 'Orange', 'Telefónica', 'Turkcell', 'Swisscom', 'Vodafone'
  ],
  research: [
    'Khalifa University', 'Universitat Pompeu Fabra', 'University of Texas', "Queen's University"
  ],
  industry: [
    'GSMA', 'Huawei GTS', 'The Linux Foundation', 'NetoAI', 'Datumo', 'Adaptive-AI'
  ],
};

// Simplified entry type for homepage TCI preview
interface TCIPreviewEntry {
  rank: number;
  provider: string;
  model: string;
  tci: number;
}


// Hero Section
function HeroSection(): JSX.Element {
  return (
    <section className={styles.hero}>
      <div className={styles.heroContent}>
        <h1 className={styles.heroTitle}>
          Evaluating AI as Network Engineers.
        </h1>
        <p className={styles.heroSubtitle}>
          The industry-standard benchmark suite for evaluating language models on telecom-specific tasks.
          Built by GSMA with leading operators and research institutions.
        </p>
        <div className={styles.heroCtas}>
          <Link to="/leaderboard" className={styles.ctaPrimary}>
            View Leaderboard
          </Link>
          <Link to="/research/dashboard" className={styles.ctaSecondary}>
            Explore Research
          </Link>
        </div>
      </div>
    </section>
  );
}

// Stats Section
function StatsSection(): JSX.Element {
  const stats = [
    { number: '4', label: 'Models Ranked', link: '/leaderboard' },
    { number: '4', label: 'Benchmarks', link: '/research/dashboard' },
    { number: '15+', label: 'Collaborators' },
    { number: '1.3K+', label: 'Eval Questions' },
  ];

  return (
    <section className={styles.stats}>
      <div className={styles.statsGrid}>
        {stats.map((stat, idx) => (
          <div key={idx} className={styles.statItem}>
            {stat.link ? (
              <Link to={stat.link} className={styles.statLink}>
                <div className={styles.statNumber}>{stat.number}</div>
                <div className={styles.statLabel}>{stat.label}</div>
              </Link>
            ) : (
              <>
                <div className={styles.statNumber}>{stat.number}</div>
                <div className={styles.statLabel}>{stat.label}</div>
              </>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

// TCI Preview Section
function TCIPreviewSection(): JSX.Element {
  const { data: rawData, loading } = useLeaderboardData();

  const leaderboard = useMemo((): TCIPreviewEntry[] => {
    return rawData
      .map(entry => {
        const tci = calculateTCI(entry);
        if (tci === null) return null;
        return {
          rank: entry.rank,
          provider: entry.provider,
          model: entry.model,
          tci,
        };
      })
      .filter((e): e is TCIPreviewEntry => e !== null)
      .sort((a, b) => b.tci - a.tci)
      .slice(0, 5);
  }, [rawData]);

  return (
    <section className={styles.tciPreview}>
      <div className={styles.sectionContainer}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Telco Capability Index (TCI)</h2>
          <p className={styles.sectionSubtitle}>
            A unified measure of AI model performance across telecommunications-specific tasks,
            using IRT-inspired methodology for meaningful cross-model comparisons.
          </p>
        </div>

        <div className={styles.tciContent}>
          <div className={styles.tciTable}>
            <div className={styles.tableHeader}>
              <span>Rank</span>
              <span>Model</span>
              <span>Provider</span>
              <span>TCI Score</span>
            </div>
            {loading ? (
              <div className={styles.tableLoading}>Loading...</div>
            ) : (
              leaderboard.map((entry, idx) => (
                <div key={entry.model} className={styles.tableRow}>
                  <span className={styles.rank}>{idx + 1}</span>
                  <span className={styles.model}>{entry.model}</span>
                  <span className={styles.provider}>{entry.provider}</span>
                  <span className={styles.tciScore}>{entry.tci}</span>
                </div>
              ))
            )}
          </div>

          <div className={styles.tciCta}>
            <Link to="/research/dashboard" className={styles.ctaSecondary}>
              View Full Dashboard
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

// Benchmarks Overview Section
function BenchmarksSection(): JSX.Element {
  return (
    <section className={styles.benchmarks}>
      <div className={styles.sectionContainer}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Benchmark Suite</h2>
          <p className={styles.sectionSubtitle}>
            Four specialized evaluations covering knowledge, reasoning, operations, and standards compliance.
          </p>
        </div>

        <div className={styles.benchmarkGrid}>
          {BENCHMARKS.map((benchmark) => (
            <div key={benchmark.key} className={styles.benchmarkCard}>
              <div className={styles.benchmarkHeader}>
                <h3 className={styles.benchmarkTitle}>{benchmark.title}</h3>
                {benchmark.difficulty && (
                  <span
                    className={styles.difficultyBadge}
                    style={{ backgroundColor: getDifficultyColor(benchmark.difficulty) }}
                  >
                    {benchmark.difficulty}
                  </span>
                )}
              </div>
              <p className={styles.benchmarkDesc}>{benchmark.description}</p>
              <div className={styles.benchmarkMeta}>
                {benchmark.samples && (
                  <span className={styles.sampleCount}>{benchmark.samples} samples</span>
                )}
                <div className={styles.benchmarkLinks}>
                  {benchmark.paperLink && (
                    <a href={benchmark.paperLink} target="_blank" rel="noopener noreferrer">
                      Paper
                    </a>
                  )}
                  {benchmark.datasetLink && (
                    <a href={benchmark.datasetLink} target="_blank" rel="noopener noreferrer">
                      Dataset
                    </a>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className={styles.benchmarksCta}>
          <Link to="/research/dashboard" className={styles.ctaSecondary}>
            Learn More About Benchmarks
          </Link>
        </div>
      </div>
    </section>
  );
}

// Collaborators Section
function CollaboratorsSection(): JSX.Element {
  return (
    <section className={styles.collaborators}>
      <div className={styles.sectionContainer}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Built with Industry Leaders</h2>
          <p className={styles.sectionSubtitle}>
            Open Telco is developed in collaboration with major telecommunications operators,
            research institutions, and industry partners.
          </p>
        </div>

        <div className={styles.collaboratorGroups}>
          <div className={styles.collaboratorGroup}>
            <h3 className={styles.groupTitle}>Operators</h3>
            <div className={styles.collaboratorList}>
              {COLLABORATORS.operators.map((name) => (
                <span key={name} className={styles.collaboratorName}>{name}</span>
              ))}
            </div>
          </div>

          <div className={styles.collaboratorGroup}>
            <h3 className={styles.groupTitle}>Research Institutions</h3>
            <div className={styles.collaboratorList}>
              {COLLABORATORS.research.map((name) => (
                <span key={name} className={styles.collaboratorName}>{name}</span>
              ))}
            </div>
          </div>

          <div className={styles.collaboratorGroup}>
            <h3 className={styles.groupTitle}>Industry Partners</h3>
            <div className={styles.collaboratorList}>
              {COLLABORATORS.industry.map((name) => (
                <span key={name} className={styles.collaboratorName}>{name}</span>
              ))}
            </div>
          </div>
        </div>

        <div className={styles.collaboratorsCta}>
          <Link to="https://github.com/gsma-research/open_telco" className={styles.ctaSecondary}>
            Join the Community
          </Link>
        </div>
      </div>
    </section>
  );
}

// Get Started Section
function GetStartedSection(): JSX.Element {
  return (
    <section className={styles.getStarted}>
      <div className={styles.sectionContainer}>
        <div className={styles.getStartedContent}>
          <h2 className={styles.getStartedTitle}>Ready to benchmark your model?</h2>
          <p className={styles.getStartedText}>
            Get started with Open Telco in minutes. Our framework supports all major model providers
            and integrates seamlessly with Inspect AI.
          </p>
          <div className={styles.codeBlock}>
            <code>
              <span className={styles.comment}># Install and run your first evaluation</span>
              {'\n'}git clone https://github.com/gsma-research/open_telco.git
              {'\n'}cd open_telco && uv sync
              {'\n'}uv run inspect eval open_telco/teleqna
            </code>
          </div>
          <div className={styles.getStartedCtas}>
            <Link to="/docs" className={styles.ctaPrimary}>
              Read the Documentation
            </Link>
            <a
              href="https://github.com/gsma-research/open_telco"
              target="_blank"
              rel="noopener noreferrer"
              className={styles.ctaSecondary}
            >
              View on GitHub
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}

// Main Homepage Component
export default function Home(): JSX.Element {
  return (
    <Layout
      title="Open Telco - AI Benchmarks for Telecommunications"
      description="GSMA's industry-standard benchmark suite for evaluating language models on telecom-specific tasks. Measure reasoning, troubleshooting, and network management capabilities."
    >
      <main>
        <HeroSection />
        <StatsSection />
        <TCIPreviewSection />
        <BenchmarksSection />
        <CollaboratorsSection />
        <GetStartedSection />
      </main>
    </Layout>
  );
}
