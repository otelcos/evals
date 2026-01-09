# Open Telco Website Architecture

This document provides comprehensive documentation for the Open Telco website, including architecture, components, data flow, and deployment information.

## Table of Contents

1. [Overview](#overview)
2. [Technology Stack](#technology-stack)
3. [Project Structure](#project-structure)
4. [Architecture Diagram](#architecture-diagram)
5. [Components](#components)
6. [Pages & Routes](#pages--routes)
7. [Data Flow](#data-flow)
8. [Styling](#styling)
9. [Build & Deployment](#build--deployment)

---

## Overview

The Open Telco website is a documentation and research portal for telecommunications LLM benchmarking. It features:

- **Research Dashboard**: Interactive visualizations of the Telco Capability Index (TCI)
- **Benchmarks Directory**: Catalog of 4 telecommunications-specific evaluations
- **Models Explorer**: Searchable/filterable list of evaluated LLMs
- **Leaderboard**: Rankings based on benchmark performance
- **User Guide**: Documentation for running evaluations
- **Notebooks**: Jupyter notebook tutorials

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Framework | [Docusaurus 3.x](https://docusaurus.io) | Static site generator |
| UI Library | React 18 | Component-based UI |
| Language | TypeScript | Type-safe JavaScript |
| Charts | [Recharts](https://recharts.org) | Data visualization |
| Styling | CSS (Infima) | Docusaurus CSS framework |
| Build | Webpack | Module bundling |
| Hosting | GitHub Pages | Static hosting |

---

## Project Structure

The website uses a **tab-based organization** where each navigation tab has its own folder containing all related content, components, and data.

```
website/
├── docusaurus.config.ts    # Main configuration (multi-instance docs)
├── sidebars.ts             # Sidebar navigation config
├── package.json            # Dependencies
├── tsconfig.json           # TypeScript config
│
├── tabs/                   # Tab-based content organization
│   ├── research/           # Research dropdown tab
│   │   ├── components/     # React components for research pages
│   │   │   ├── TelcoCapabilityIndex.tsx  # Main TCI chart
│   │   │   ├── ModelsPage.tsx            # Models explorer
│   │   │   └── BenchmarkCard.tsx         # Benchmark info card
│   │   └── docs/
│   │       ├── dashboard.md    # TCI visualization page
│   │       ├── benchmarks.md   # Benchmark catalog
│   │       └── models.md       # Models explorer page
│   │
│   ├── leaderboard/        # Leaderboard tab
│   │   ├── data/
│   │   │   └── leaderboard.json  # Single source of truth
│   │   └── docs/
│   │       └── leaderboard.md  # Full leaderboard table
│   │
│   ├── user-guide/         # User Guide tab (main docs with sidebar)
│   │   └── docs/
│   │       ├── welcome.md      # Homepage
│   │       └── contributing.md # Contribution guide
│   │
│   ├── notebooks/          # Notebooks tab
│   │   └── docs/
│   │       ├── how-to-use.md
│   │       ├── implementing-evals.md
│   │       └── network-environments.md
│   │
│   ├── reference/          # Reference tab
│   │   └── docs/
│   │       └── reference.md
│   │
│   ├── community/          # Community tab
│   │   └── docs/
│   │       └── community.md
│   │
│   └── resources/          # Resources (footer section)
│       └── docs/
│           ├── datasets.md
│           └── agents.md
│
├── src/
│   └── css/
│       └── custom.css      # Global custom styling
│
├── static/                 # Static assets
│   └── img/                # Images and favicon
│
└── build/                  # Generated static site (git-ignored)
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OPEN TELCO WEBSITE                                 │
│                         (Docusaurus Static Site)                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              NAVIGATION                                      │
├─────────────┬─────────────┬──────────────┬──────────────┬──────────────────┤
│  Research ▼ │ Leaderboard │  User Guide  │  Reference   │    Notebooks     │
├─────────────┴─────────────┴──────────────┴──────────────┴──────────────────┤
│  ├─ Dashboard                                                               │
│  ├─ Benchmarks                                                              │
│  └─ Models                                                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          TAB-BASED ORGANIZATION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   tabs/research/               tabs/leaderboard/      tabs/user-guide/      │
│   ├── components/              ├── data/              └── docs/             │
│   │   ├── TelcoCapability...   │   └── *.csv              ├── welcome.md    │
│   │   ├── ModelsPage.tsx       └── docs/                  └── contributing  │
│   │   └── BenchmarkCard.tsx        └── leaderboard.md                       │
│   └── docs/                                                                  │
│       ├── dashboard.md         tabs/notebooks/        tabs/resources/       │
│       ├── benchmarks.md        └── docs/              └── docs/             │
│       └── models.md                ├── how-to-use.md      ├── datasets.md   │
│                                    └── ...                └── agents.md     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   tabs/leaderboard/data/leaderboard.json                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ [                                                                    │   │
│   │   { "rank": 1, "model": "gpt-5.2", "provider": "OpenAI",            │   │
│   │     "teleqna": 83.6, "telelogs": 75.0, "telemath": 39.0, "tsg": 54 }│   │
│   │   ...                                                                │   │
│   │ ]                                                                    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Data Flow:                                                                 │
│   1. JSON file served from /open_telco/data/leaderboard.json                │
│   2. React components fetch JSON on mount                                    │
│   3. parseJSON() maps fields to LeaderboardEntry type                       │
│   4. calculateTCI() computes capability index using IRT methodology         │
│   5. Components render filtered/sorted data                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        BUILD & DEPLOYMENT                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   npm run build                                                              │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐ │
│   │  Docusaurus │───▶│   Webpack   │───▶│  /build/ (static HTML/JS/CSS)   │ │
│   │   Compiler  │    │   Bundler   │    │                                  │ │
│   └─────────────┘    └─────────────┘    └─────────────────────────────────┘ │
│                                                      │                       │
│                                                      ▼                       │
│                              ┌────────────────────────────────────────────┐ │
│                              │         GitHub Pages                        │ │
│                              │  https://gsma-research.github.io/open_telco│ │
│                              └────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### TelcoCapabilityIndex

**File**: `tabs/research/components/TelcoCapabilityIndex.tsx`

The main visualization component displaying the Telco Capability Index chart, inspired by [Epoch AI's Capability Index](https://epoch.ai/data/eci).

#### Features
- Interactive scatter plot using Recharts
- Dynamic Y-axis scaling based on actual TCI scores
- Organization color coding (16 predefined colors)
- Hover tooltips showing model details and benchmark scores
- Labeled key models (GPT-5, Claude-Sonnet-4.5, etc.)

#### Key Functions

```typescript
// Parse CSV with quoted field handling
function parseCSV(csvText: string): LeaderboardEntry[]

// Calculate TCI using Item Response Theory (IRT)
function calculateTCI(entry: LeaderboardEntry): number | null
```

#### TCI Calculation

The TCI score is computed using an IRT-inspired methodology:

```
P(score | θ, β, α) = σ(α(θ - β))
```

Where:
- **θ (theta)**: Model capability
- **β (beta)**: Benchmark difficulty
- **α (alpha)**: Discrimination slope

Benchmark difficulties (estimated):
| Benchmark | Difficulty | Slope |
|-----------|------------|-------|
| TeleQnA | 0.7 (easier) | 1.2 |
| TeleLogs | 0.3 (harder) | 1.5 |
| TeleMath | 0.4 (medium) | 1.3 |
| 3GPP-TSG | 0.4 (medium) | 1.2 |

---

### ModelsPage

**File**: `tabs/research/components/ModelsPage.tsx`

A searchable, filterable explorer for all evaluated models.

#### Features
- Search by model name or provider
- Sort by TCI, Mean Score, or Name
- Filter by provider (checkbox multi-select)
- TCI score badges with color coding
- Responsive two-column layout

#### State Management
```typescript
const [searchQuery, setSearchQuery] = useState('');
const [sortBy, setSortBy] = useState<'tci' | 'mean' | 'name'>('tci');
const [selectedProviders, setSelectedProviders] = useState<Set<string>>(new Set());
```

---

### BenchmarkCard

**File**: `tabs/research/components/BenchmarkCard.tsx`

A reusable card component for displaying benchmark information.

#### Props
```typescript
interface BenchmarkCardProps {
  title: string;
  description: string;
  link?: string;
  paperLink?: string;
  datasetLink?: string;
}
```

---

## Pages & Routes

| Route | File | Description |
|-------|------|-------------|
| `/` | `tabs/user-guide/docs/welcome.md` | Homepage |
| `/research/dashboard` | `tabs/research/docs/dashboard.md` | TCI visualization |
| `/research/benchmarks` | `tabs/research/docs/benchmarks.md` | Benchmark catalog |
| `/research/models` | `tabs/research/docs/models.md` | Models explorer |
| `/leaderboard` | `tabs/leaderboard/docs/leaderboard.md` | Full leaderboard table |
| `/reference` | `tabs/reference/docs/reference.md` | API reference |
| `/notebooks/how-to-use` | `tabs/notebooks/docs/how-to-use.md` | Usage guide |
| `/community` | `tabs/community/docs/community.md` | Community links |
| `/resources/datasets` | `tabs/resources/docs/datasets.md` | Dataset information |
| `/resources/agents` | `tabs/resources/docs/agents.md` | Telco agents information |

### Navigation Structure

```
Navbar:
├── Research (dropdown)
│   ├── Dashboard
│   ├── Benchmarks
│   └── Models
├── Leaderboard
├── User Guide (sidebar)
├── Reference
├── Notebooks
├── Community
└── GitHub (external)
```

---

## Data Flow

### JSON Loading

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────┐
│   Component  │────▶│    fetch()   │────▶│  /data/leaderboard.json      │
│    Mount     │     │              │     │                               │
└──────────────┘     └──────────────┘     └──────────────────────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ parseJSON()  │
                     │  - Map to    │
                     │    type      │
                     │  - Extract   │
                     │    scores    │
                     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │calculateTCI()│
                     │  - IRT model │
                     │  - Weighted  │
                     │    average   │
                     │  - Scale to  │
                     │    90-150    │
                     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │    Render    │
                     │  - Charts    │
                     │  - Lists     │
                     │  - Cards     │
                     └──────────────┘
```

### Data File Format

`tabs/leaderboard/data/leaderboard.json`:
```json
[
  {
    "rank": 1,
    "model": "gpt-5.2",
    "provider": "OpenAI",
    "repo": "openrouter/openai/gpt-5.2",
    "mean": 62.9,
    "teleqna": 83.6,
    "teleqna_stderr": 1.17,
    "telelogs": 75.0,
    "telelogs_stderr": 4.35,
    "telemath": 39.0,
    "telemath_stderr": 4.9,
    "tsg": 54.0,
    "tsg_stderr": 5.01
  }
]
```

---

## Styling

### Brand Colors

| Color | Hex | Usage |
|-------|-----|-------|
| GSMA Red | `#E60000` | Primary, links, active states |
| Dark Red | `#cf0000` | Hover states |
| Teal | `#008080` | Benchmark titles, chart accents |

### Organization Colors (Charts)

```typescript
const COLORS: Record<string, string> = {
  'Google': '#4DB6AC',
  'OpenAI': '#F48FB1',
  'Meta': '#FFAB91',
  'Anthropic': '#B39DDB',
  'Grok': '#5C6BC0',
  'Qwen': '#81C784',
  'Mistral': '#FF8A65',
  // ...
};
```

### CSS Classes

| Class | Purpose |
|-------|---------|
| `.research-tabs` | Epoch AI-style pill tab container |
| `.research-tab` | Individual tab button |
| `.research-tab.active` | Active tab state |
| `.benchmark-grid` | Responsive card grid |

---

## Build & Deployment

### Development

```bash
cd website
npm install
npm run start     # Local dev server at localhost:3000
```

### Production Build

```bash
npm run build     # Generates /build directory
npm run serve     # Preview production build
```

### Deployment (GitHub Pages)

The site is deployed via GitHub Actions to:
`https://gsma-research.github.io/open_telco/`

Configuration in `docusaurus.config.ts`:
```typescript
url: 'https://gsma-research.github.io',
baseUrl: '/open_telco/',
organizationName: 'gsma-research',
projectName: 'open_telco',
```

---

## Configuration Files

### docusaurus.config.ts

Key settings:
```typescript
{
  title: 'Open Telco',
  tagline: 'A collection of telco evals...',
  baseUrl: '/open_telco/',

  // Tab-based organization using multi-instance docs
  staticDirectories: ['static', 'tabs/leaderboard'],
  plugins: [
    // Research, Leaderboard, Notebooks, Reference, Community, Resources
    // Each as separate docs instance
  ],

  docs: {
    path: 'tabs/user-guide/docs',
    routeBasePath: '/',  // User Guide at root
  },
}
```

### sidebars.ts

Sidebar navigation structure (User Guide only):
```typescript
{
  docsSidebar: [
    { type: 'category', label: 'Open Telco', items: ['welcome', 'contributing'] },
  ],
}
```

---

## Adding New Content

### New Benchmark

1. Add card to `tabs/research/docs/benchmarks.md`:
```jsx
<BenchmarkCard
  title="NewBenchmark"
  description="..."
  paperLink="https://arxiv.org/..."
  datasetLink="https://huggingface.co/..."
/>
```

2. Update CSV with new column
3. Modify `parseCSV()` and `calculateTCI()` if needed

### New Model Provider

1. Add color to `COLORS` object in both:
   - `tabs/research/components/TelcoCapabilityIndex.tsx`
   - `tabs/research/components/ModelsPage.tsx`

### New Documentation Page

1. Create `.md` file in appropriate `tabs/*/docs/` folder
2. Add to `sidebars.ts` if needed (for User Guide pages)
3. Update navbar in `docusaurus.config.ts` if top-level

### New Tab Section

1. Create folder under `tabs/new-section/docs/`
2. Add plugin instance in `docusaurus.config.ts`:
```typescript
plugins: [
  // ... existing plugins
  [
    '@docusaurus/plugin-content-docs',
    {
      id: 'new-section',
      path: 'tabs/new-section/docs',
      routeBasePath: 'new-section',
      sidebarPath: false,
    },
  ],
],
```
3. Add navbar item in `docusaurus.config.ts`

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| JSON not loading | Check path `/open_telco/data/leaderboard.json` |
| Chart not rendering | Verify Recharts imported |
| Styles not applying | Check CSS class names |
| Build fails on LaTeX | Use code blocks instead of math |
| Component not found | Check import path uses `@site/tabs/...` |

### Debug Mode

```bash
DEBUG=* npm run build
```

---

## License

Part of the GSMA Open Telco project. See root LICENSE file.
