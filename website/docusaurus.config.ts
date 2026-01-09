import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Open Telco',
  tagline: 'A collection of telco evals for the next generation of connectivity.',
  favicon: 'img/favicon.svg',

  future: {
    v4: true,
  },

  // GitHub Pages deployment config
  url: 'https://gsma-research.github.io',
  baseUrl: '/open_telco/',

  organizationName: 'gsma-research',
  projectName: 'open_telco',
  trailingSlash: false,

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // Static directories - serve leaderboard data from tabs/leaderboard
  staticDirectories: ['static', 'tabs/leaderboard'],

  // Multi-instance docs plugins for tab-based organization
  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'research',
        path: 'tabs/research/docs',
        routeBasePath: 'research',
        sidebarPath: false,
        editUrl: 'https://github.com/gsma-research/open_telco/tree/main/website/',
      },
    ],
    // Leaderboard now uses custom React page at src/pages/leaderboard.tsx
    // Markdown docs moved to /leaderboard/details for reference
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'leaderboard',
        path: 'tabs/leaderboard/docs',
        routeBasePath: 'leaderboard/details',
        sidebarPath: false,
        editUrl: 'https://github.com/gsma-research/open_telco/tree/main/website/',
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'resources',
        path: 'tabs/resources/docs',
        routeBasePath: 'resources',
        sidebarPath: false,
        editUrl: 'https://github.com/gsma-research/open_telco/tree/main/website/',
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'blog',
        path: 'tabs/blog/docs',
        routeBasePath: 'blog',
        sidebarPath: false,
        editUrl: 'https://github.com/gsma-research/open_telco/tree/main/website/',
      },
    ],
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          path: 'tabs/user-guide/docs',
          sidebarPath: './sidebars.ts',
          routeBasePath: '/docs',
          editUrl: 'https://github.com/gsma-research/open_telco/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/open-telco-social-card.jpg',
    colorMode: {
      defaultMode: 'light',
      disableSwitch: true,
      respectPrefersColorScheme: false,
    },
    navbar: {
      title: 'Open Telco',
      logo: {
        alt: 'GSMA Logo',
        src: 'img/GSMA.jpeg',
        height: 28,
      },
      items: [
        {
          type: 'dropdown',
          label: 'Research',
          position: 'left',
          items: [
            {
              to: '/research/dashboard',
              label: 'Dashboard',
            },
            {
              to: '/research/benchmarks',
              label: 'Benchmarks',
            },
            {
              to: '/research/models',
              label: 'Models',
            },
          ],
        },
        {
          to: '/leaderboard',
          label: 'Leaderboard',
          position: 'left',
        },
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          to: '/blog',
          label: 'Blog',
          position: 'left',
        },
        {
          href: 'https://github.com/gsma-research/open_telco',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'light',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'User Guide',
              to: '/docs',
            },
            {
              label: 'Getting Started',
              to: '/docs/installation',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Datasets & Methodology',
              to: '/resources/datasets',
            },
            {
              label: 'Telco Specific Agents',
              to: '/resources/agents',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/gsma-research/open_telco',
            },
            {
              label: 'GSMA',
              href: 'https://www.gsma.com',
            },
            {
              label: 'Hugging Face',
              href: 'https://huggingface.co/datasets/GSMA/open_telco',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} GSMA. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 3,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
