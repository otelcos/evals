---
id: faq
title: FAQ
sidebar_label: FAQ
sidebar_position: 5
---

import FAQ from '@site/tabs/user-guide/components/FAQ';

# Frequently Asked Questions

Common questions about Open Telco, organized by topic.

## Getting Started

<FAQ items={[
  {
    question: "What models can I use with Open Telco?",
    answer: (
      <>
        <p>Open Telco supports any model available through <a href="https://inspect.aisi.org.uk/models.html">Inspect AI</a>:</p>
        <ul>
          <li><strong>OpenAI:</strong> gpt-4o, gpt-4-turbo, gpt-4o-mini</li>
          <li><strong>Anthropic:</strong> claude-sonnet-4-20250514, claude-3-5-haiku-latest</li>
          <li><strong>OpenRouter:</strong> Access to 100+ models including Gemini, Llama, Mistral</li>
          <li><strong>Local:</strong> Ollama, vLLM for self-hosted models</li>
        </ul>
        <p>Model format: <code>provider/model-name</code> (e.g., <code>openai/gpt-4o</code>)</p>
      </>
    )
  },
  {
    question: "How do I get a HuggingFace token?",
    answer: (
      <>
        <ol>
          <li>Create an account at <a href="https://huggingface.co/join">huggingface.co</a></li>
          <li>Go to <a href="https://huggingface.co/settings/tokens">Settings &gt; Access Tokens</a></li>
          <li>Click "New token" and create a token with read access</li>
          <li>Add it to your <code>.env</code> file as <code>HF_TOKEN=your_token_here</code></li>
        </ol>
      </>
    )
  },
  {
    question: "Which benchmark should I start with?",
    answer: (
      <>
        <p><strong>TeleQnA</strong> is the best starting point:</p>
        <ul>
          <li>Fastest to run (simple Q&A format)</li>
          <li>Broad coverage of telecom topics</li>
          <li>Good baseline for model comparison</li>
        </ul>
        <p>Run it with: <code>uv run inspect eval src/open_telco/teleqna/teleqna.py --model openai/gpt-4o --limit 50</code></p>
      </>
    )
  },
  {
    question: "How much does it cost to run evaluations?",
    answer: (
      <>
        <p>Costs depend on the model and benchmark size:</p>
        <ul>
          <li><strong>TeleQnA (full):</strong> ~10K API calls, varies by model pricing</li>
          <li><strong>TeleMath:</strong> ~500 calls but longer prompts</li>
          <li><strong>Testing:</strong> Use <code>--limit 20</code> to test cheaply</li>
        </ul>
        <p>Tip: Use <code>gpt-4o-mini</code> for initial testing, then switch to full models for final runs.</p>
      </>
    )
  }
]} />

## Understanding Benchmarks

<FAQ items={[
  {
    question: "What does each benchmark measure?",
    answer: (
      <>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
              <th style={{ textAlign: 'left', padding: '8px' }}>Benchmark</th>
              <th style={{ textAlign: 'left', padding: '8px' }}>Measures</th>
              <th style={{ textAlign: 'left', padding: '8px' }}>Use Case</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #e5e7eb' }}>
              <td style={{ padding: '8px' }}><strong>TeleQnA</strong></td>
              <td style={{ padding: '8px' }}>General telecom knowledge</td>
              <td style={{ padding: '8px' }}>Baseline capability</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #e5e7eb' }}>
              <td style={{ padding: '8px' }}><strong>TeleMath</strong></td>
              <td style={{ padding: '8px' }}>Mathematical reasoning</td>
              <td style={{ padding: '8px' }}>Signal processing, optimization</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #e5e7eb' }}>
              <td style={{ padding: '8px' }}><strong>TeleLogs</strong></td>
              <td style={{ padding: '8px' }}>Root cause analysis</td>
              <td style={{ padding: '8px' }}>Network operations</td>
            </tr>
            <tr>
              <td style={{ padding: '8px' }}><strong>3GPP TSG</strong></td>
              <td style={{ padding: '8px' }}>Standards understanding</td>
              <td style={{ padding: '8px' }}>Technical spec work</td>
            </tr>
          </tbody>
        </table>
      </>
    )
  },
  {
    question: "How do I interpret accuracy scores?",
    answer: (
      <>
        <p>Scores vary significantly by benchmark complexity:</p>
        <ul>
          <li><strong>TeleQnA:</strong> 40-70% typical (multiple choice)</li>
          <li><strong>TeleLogs:</strong> 30-60% typical (requires reasoning)</li>
          <li><strong>TeleMath:</strong> 20-50% typical (multi-step calculations)</li>
          <li><strong>3GPP TSG:</strong> 50-80% typical (classification)</li>
        </ul>
        <p>Higher isn't always meaningful across benchmarks. Compare models on the same benchmark for fair comparison.</p>
      </>
    )
  },
  {
    question: "Why are my results inconsistent between runs?",
    answer: (
      <>
        <p>LLM outputs have inherent randomness. To get stable results:</p>
        <ul>
          <li>Run multiple epochs: <code>--epochs 3</code></li>
          <li>Use more samples: <code>--limit 100</code> instead of <code>--limit 10</code></li>
          <li>Set temperature to 0 if the model supports it</li>
        </ul>
        <p>The final score will be averaged across epochs.</p>
      </>
    )
  }
]} />

## Running Evaluations

<FAQ items={[
  {
    question: "How long do evaluations take?",
    answer: (
      <>
        <p>Depends on benchmark, model, and sample count:</p>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
              <th style={{ textAlign: 'left', padding: '8px' }}>Benchmark</th>
              <th style={{ textAlign: 'left', padding: '8px' }}>50 samples</th>
              <th style={{ textAlign: 'left', padding: '8px' }}>Full dataset</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #e5e7eb' }}>
              <td style={{ padding: '8px' }}>TeleQnA</td>
              <td style={{ padding: '8px' }}>~2-5 min</td>
              <td style={{ padding: '8px' }}>~30-60 min</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #e5e7eb' }}>
              <td style={{ padding: '8px' }}>TeleLogs</td>
              <td style={{ padding: '8px' }}>~5-10 min</td>
              <td style={{ padding: '8px' }}>~1-2 hours</td>
            </tr>
            <tr>
              <td style={{ padding: '8px' }}>TeleMath</td>
              <td style={{ padding: '8px' }}>~10-20 min</td>
              <td style={{ padding: '8px' }}>~2-4 hours</td>
            </tr>
          </tbody>
        </table>
      </>
    )
  },
  {
    question: "How do I run multiple models at once?",
    answer: (
      <>
        <p>Use <code>inspect eval-set</code> with comma-separated models:</p>
        <pre style={{ background: '#f5f5f5', padding: '12px', borderRadius: '4px', overflow: 'auto' }}>
{`uv run inspect eval-set src/open_telco/teleqna/teleqna.py \\
  --model openai/gpt-4o,anthropic/claude-sonnet-4-20250514 \\
  --log-dir logs/comparison`}
        </pre>
        <p>Models run in parallel, and results are saved to the same log directory for easy comparison.</p>
      </>
    )
  },
  {
    question: "Can I resume an interrupted evaluation?",
    answer: (
      <>
        <p>Yes! Re-run the same command with the same <code>--log-dir</code>:</p>
        <pre style={{ background: '#f5f5f5', padding: '12px', borderRadius: '4px', overflow: 'auto' }}>
{`uv run inspect eval-set src/open_telco/teleqna/teleqna.py \\
  --model openai/gpt-4o \\
  --log-dir logs/my-run`}
        </pre>
        <p>Inspect will pick up where it left off, skipping completed samples.</p>
      </>
    )
  }
]} />

## Troubleshooting

<FAQ items={[
  {
    question: "I'm getting 'HF_TOKEN not set' errors",
    answer: (
      <>
        <p>Check these common issues:</p>
        <ul>
          <li>Is your <code>.env</code> file in the project root (not a subdirectory)?</li>
          <li>Does your token have read access on HuggingFace?</li>
          <li>Try: <code>source .env</code> then run the command again</li>
          <li>Check for spaces around <code>=</code> (should be <code>HF_TOKEN=abc</code> not <code>HF_TOKEN = abc</code>)</li>
        </ul>
      </>
    )
  },
  {
    question: "I'm getting rate limit errors",
    answer: (
      <>
        <p>Add retry options to handle rate limits gracefully:</p>
        <pre style={{ background: '#f5f5f5', padding: '12px', borderRadius: '4px', overflow: 'auto' }}>
{`uv run inspect eval src/open_telco/teleqna/teleqna.py \\
  --model openai/gpt-4o \\
  --retry-attempts 5 \\
  --retry-wait 60`}
        </pre>
        <p>This retries failed requests up to 5 times with exponential backoff starting at 60 seconds.</p>
      </>
    )
  },
  {
    question: "Evaluations are running very slowly",
    answer: (
      <>
        <p>Try these optimizations:</p>
        <ul>
          <li>Use <code>--limit 10</code> for initial testing</li>
          <li>Try faster models like <code>gpt-4o-mini</code></li>
          <li>Check your internet connection</li>
          <li>Consider using <code>--max-tasks</code> to control parallelism</li>
        </ul>
      </>
    )
  },
  {
    question: "Model not found errors",
    answer: (
      <>
        <p>Check these:</p>
        <ul>
          <li>Model format should be <code>provider/model-name</code></li>
          <li>Verify your API key is set for that provider</li>
          <li>Check <a href="https://inspect.aisi.org.uk/models.html">Inspect AI docs</a> for exact model names</li>
        </ul>
        <p>Common examples:</p>
        <ul>
          <li><code>openai/gpt-4o</code> (not <code>gpt-4o</code>)</li>
          <li><code>anthropic/claude-sonnet-4-20250514</code></li>
          <li><code>openrouter/google/gemini-2.0-flash-001</code></li>
        </ul>
      </>
    )
  }
]} />

## Getting Help

- **GitHub Issues:** [github.com/gsma-research/open_telco/issues](https://github.com/gsma-research/open_telco/issues)
- **Inspect AI Docs:** [inspect.aisi.org.uk](https://inspect.aisi.org.uk/)
- **Email:** [emolero@gsma.com](mailto:emolero@gsma.com)
