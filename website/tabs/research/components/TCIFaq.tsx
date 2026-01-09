import React from 'react';
import { useAccordion } from '../../../src/hooks/useAccordion';

interface FAQItem {
  question: string;
  answer: React.ReactNode;
}

const FAQ_ITEMS: FAQItem[] = [
  {
    question: "What does the TCI represent?",
    answer: (
      <>
        <p>
          The Telco Capability Index (TCI) is a unified measure of AI model performance
          across telecommunications-specific tasks. Unlike simple averages, TCI uses
          Item Response Theory (IRT) principles to provide meaningful cross-model
          comparisons that account for benchmark difficulty and discriminative power.
        </p>
        <p>
          Think of it like an IQ score for telecommunications capability—a single number
          that captures broad, underlying ability rather than performance on any individual task.
        </p>
      </>
    ),
  },
  {
    question: "How is TCI calculated?",
    answer: (
      <>
        <p>
          The technical foundation comes from Item Response Theory (IRT), a statistical
          framework originally developed for educational testing. The core model uses a
          logistic function:
        </p>
        <p style={{ fontFamily: 'monospace', background: '#f5f5f5', padding: '12px', borderRadius: '4px' }}>
          P(score | θ, β, α) = σ(α(θ - β))
        </p>
        <p>Where:</p>
        <ul>
          <li><strong>θ (theta)</strong> represents the model's capability</li>
          <li><strong>β (beta)</strong> represents the benchmark's difficulty</li>
          <li><strong>α (alpha)</strong> is a slope parameter related to the distribution of difficulty across questions</li>
        </ul>
        <p>
          Higher scores on harder benchmarks contribute more to the final capability estimate.
          For example, strong performance on TeleLogs (which has lower average scores) indicates
          higher capability than equivalent performance on TeleQnA.
        </p>
      </>
    ),
  },
  {
    question: "How should I interpret TCI values?",
    answer: (
      <>
        <p>
          TCI values are scaled for intuitive interpretation:
        </p>
        <ul>
          <li><strong>Below 110:</strong> Below average telecommunications capability</li>
          <li><strong>110-120:</strong> Average performance</li>
          <li><strong>120-130:</strong> Above average capability</li>
          <li><strong>Above 130:</strong> Exceptional telecommunications domain expertise</li>
        </ul>
        <p>
          The difference between scores is meaningful—a model with TCI 125 consistently
          outperforms one with TCI 115 across harder benchmarks. Values are comparable
          like Elo scores; the scale is relative with no meaningful absolute zero.
        </p>
      </>
    ),
  },
  {
    question: "Which benchmarks are used?",
    answer: (
      <>
        <p>
          The TCI currently incorporates 4 telecommunications-specific evaluations:
        </p>
        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '12px' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
              <th style={{ textAlign: 'left', padding: '8px 12px', fontWeight: 600 }}>Benchmark</th>
              <th style={{ textAlign: 'left', padding: '8px 12px', fontWeight: 600 }}>Description</th>
              <th style={{ textAlign: 'left', padding: '8px 12px', fontWeight: 600 }}>Difficulty</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #e5e7eb' }}>
              <td style={{ padding: '8px 12px', fontWeight: 500 }}>TeleQnA</td>
              <td style={{ padding: '8px 12px', color: '#4b5563' }}>Multiple-choice questions on telecom knowledge</td>
              <td style={{ padding: '8px 12px', color: '#6b7280' }}>Medium</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #e5e7eb' }}>
              <td style={{ padding: '8px 12px', fontWeight: 500 }}>TeleLogs</td>
              <td style={{ padding: '8px 12px', color: '#4b5563' }}>Log analysis and troubleshooting</td>
              <td style={{ padding: '8px 12px', color: '#6b7280' }}>Hard</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #e5e7eb' }}>
              <td style={{ padding: '8px 12px', fontWeight: 500 }}>TeleMath</td>
              <td style={{ padding: '8px 12px', color: '#4b5563' }}>Mathematical reasoning in telecom contexts</td>
              <td style={{ padding: '8px 12px', color: '#6b7280' }}>Medium-Hard</td>
            </tr>
            <tr>
              <td style={{ padding: '8px 12px', fontWeight: 500 }}>3GPP-TSG</td>
              <td style={{ padding: '8px 12px', color: '#4b5563' }}>3GPP Technical Specification Group classification</td>
              <td style={{ padding: '8px 12px', color: '#6b7280' }}>Medium-Hard</td>
            </tr>
          </tbody>
        </table>
        <p style={{ marginTop: '12px', fontSize: '0.875rem', color: '#6b7280' }}>
          TeleYAML is excluded from TCI as it uses LLM-as-judge evaluation which introduces additional variance.
        </p>
      </>
    ),
  },
  {
    question: "Why isn't this model's TCI higher, if it leads some benchmarks?",
    answer: (
      <>
        <p>
          TCI weights harder benchmarks more heavily than easier ones. A model that
          excels at TeleQnA (our easiest benchmark) but performs poorly on TeleLogs
          (our hardest) will have a lower TCI than one with more balanced performance.
        </p>
        <p>
          This design reflects real-world requirements: a model that can handle
          difficult log analysis and mathematical reasoning is more valuable for
          telecommunications applications than one that only answers basic knowledge questions.
        </p>
      </>
    ),
  },
  {
    question: "How do you decide which benchmarks to include?",
    answer: (
      <>
        <p>
          We include benchmarks that meet several criteria:
        </p>
        <ul>
          <li>Evaluates capabilities relevant to telecommunications operators and vendors</li>
          <li>Has sufficient difficulty to discriminate between model capabilities</li>
          <li>Uses rigorous evaluation methodology (not easily gameable)</li>
          <li>Publicly available or reproducible</li>
          <li>Uses 0-1 scoring for IRT compatibility</li>
        </ul>
        <p>
          We emphasize diversity of skills tested, economic value to the industry,
          lack of saturation, and widespread adoption.
        </p>
      </>
    ),
  },
  {
    question: "Why did the TCI score of a model change?",
    answer: (
      <>
        <p>
          TCI scores can change for several reasons:
        </p>
        <ul>
          <li><strong>New evaluations:</strong> When we add results from additional benchmarks, the combined score may shift</li>
          <li><strong>Methodology updates:</strong> Improvements to our IRT model parameters may affect all scores</li>
          <li><strong>Benchmark corrections:</strong> Fixing errors in evaluation data affects dependent scores</li>
          <li><strong>Joint fitting:</strong> Because models are fitted jointly, adding new models can slightly shift existing scores</li>
        </ul>
        <p>
          We version our methodology and document significant changes. Historical scores
          are preserved in our data archives.
        </p>
      </>
    ),
  },
  {
    question: "Isn't it a problem if model developers release only their best scores?",
    answer: (
      <>
        <p>
          Selective reporting is a known issue in AI benchmarking. We mitigate this by:
        </p>
        <ul>
          <li>Running evaluations ourselves using standardized configurations</li>
          <li>Requiring evaluation logs and reproducibility documentation for submissions</li>
          <li>Flagging results that significantly diverge from our internal evaluations</li>
          <li>Using multiple benchmarks so cherry-picking single scores has limited impact on TCI</li>
        </ul>
        <p>
          Our IRT methodology is also robust to some missing data, so models without
          results on every benchmark can still receive a valid TCI score.
        </p>
      </>
    ),
  },
  {
    question: "Isn't it a problem if model developers optimize for benchmark scores?",
    answer: (
      <>
        <p>
          Benchmark optimization (teaching to the test) is a concern for all AI evaluations.
          Our benchmarks use several defenses:
        </p>
        <ul>
          <li><strong>TeleLogs:</strong> Uses realistic log formats that require understanding, not memorization</li>
          <li><strong>TeleMath:</strong> Tests reasoning steps, not just final answers</li>
          <li><strong>3GPP-TSG:</strong> Covers a broad document classification space</li>
        </ul>
        <p>
          We also periodically refresh evaluation sets and monitor for suspiciously
          large improvements that don't generalize to held-out tasks. TCI inherits
          these limitations of benchmarking but mitigates through diversity.
        </p>
      </>
    ),
  },
  {
    question: "Why isn't my model included?",
    answer: (
      <>
        <p>
          We prioritize models based on:
        </p>
        <ul>
          <li>Relevance to telecommunications industry (deployment likelihood)</li>
          <li>Availability for evaluation (API access or open weights)</li>
          <li>Community interest and research impact</li>
          <li>Minimum 3 benchmark evaluations required for valid TCI</li>
        </ul>
        <p>
          To request inclusion, submit your model via{' '}
          <a href="https://github.com/gsma-research/open_telco/issues" target="_blank" rel="noopener noreferrer">
            GitHub Issues
          </a>{' '}
          with evaluation results and methodology documentation.
        </p>
      </>
    ),
  },
  {
    question: "Why isn't my benchmark included?",
    answer: (
      <>
        <p>
          We're actively expanding our benchmark coverage. For a new benchmark to be
          included in TCI, it should:
        </p>
        <ul>
          <li>Target telecommunications-specific skills not covered by existing benchmarks</li>
          <li>Have documented evaluation methodology and scoring criteria</li>
          <li>Be reproducible by independent researchers</li>
          <li>Show meaningful variance across current frontier models</li>
        </ul>
        <p>
          Propose new benchmarks through our{' '}
          <a href="https://github.com/gsma-research/open_telco/issues" target="_blank" rel="noopener noreferrer">
            GitHub repository
          </a>. We especially welcome benchmarks covering network operations,
          customer service, and emerging 5G/6G technologies.
        </p>
      </>
    ),
  },
];

export default function TCIFaq(): JSX.Element {
  const { isOpen, handleToggle } = useAccordion();

  return (
    <div className="tci-faq-section">
      <div className="tci-faq-container">
        <div className="tci-faq-header">
          <h2>FAQ</h2>
        </div>
        <div className="tci-faq-accordion">
          {FAQ_ITEMS.map((item, index) => (
            <div
              key={index}
              className={`tci-faq-item ${isOpen(index) ? 'open' : ''}`}
            >
              <button
                className="tci-faq-question"
                onClick={() => handleToggle(index)}
                aria-expanded={isOpen(index)}
              >
                <span>{item.question}</span>
                <span className="tci-faq-icon">
                  {isOpen(index) ? '−' : '+'}
                </span>
              </button>
              <div className="tci-faq-answer">
                <div className="tci-faq-answer-content">
                  {item.answer}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
