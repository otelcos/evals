import React, { useState, useMemo } from 'react';
import BlogCard, { BlogPost } from './BlogCard';
import BlogHero from './BlogHero';
import BlogSearch from './BlogSearch';
import CategoryFilter, { Category } from './CategoryFilter';

// Sample blog posts data - Cohere style with multiple categories
const BLOG_POSTS: BlogPost[] = [
  {
    slug: '2026-01-08-teleqna-v2-release',
    title: 'TeleQnA v2.0: Expanding the Telecommunications Knowledge Benchmark',
    description: 'We are excited to announce the release of TeleQnA v2.0, featuring 2,000 new questions spanning 5G-Advanced, O-RAN, and network slicing topics.',
    categories: ['Research'],
    author: { name: 'COHERE LABS TEAM', avatar: '/img/authors/team.svg' },
    date: '2026-01-08',
    image: '/img/blog/teleqna-v2.svg',
    featured: true,
  },
  {
    slug: '2026-01-05-tci-methodology',
    title: 'Understanding the Telco Capability Index: An IRT-Based Approach to Model Evaluation',
    categories: ['Product', 'Newsroom'],
    author: { name: 'OPEN TELCO TEAM', avatar: '/img/authors/research.svg' },
    date: '2026-01-05',
    image: '/img/blog/tci-methodology.svg',
  },
  {
    slug: '2025-12-20-benchmark-results-q4',
    title: "Building trust in AI: Open Telco's approach to AI governance",
    categories: ['Research'],
    author: { name: 'OPEN TELCO TEAM', avatar: '/img/authors/research.svg' },
    date: '2025-12-20',
    image: '/img/blog/q4-results.svg',
  },
  {
    slug: '2025-12-15-getting-started-tutorial',
    title: 'Connect 2025: Why Collaboration is the Ultimate Research Superpower',
    categories: ['Research'],
    author: { name: 'COHERE LABS TEAM', avatar: '/img/authors/team.svg' },
    date: '2025-12-15',
    image: '/img/blog/getting-started.svg',
  },
  {
    slug: '2025-12-01-open-telco-launch',
    title: 'Introducing Open Telco: Benchmarking LLMs for Telecommunications',
    categories: ['Company'],
    author: { name: 'GSMA RESEARCH', avatar: '/img/authors/gsma.svg' },
    date: '2025-12-01',
    image: '/img/blog/launch.svg',
  },
  {
    slug: '2025-11-20-5g-eval-results',
    title: 'How LLMs Perform on 5G Network Configuration Tasks',
    categories: ['Research'],
    author: { name: 'RESEARCH TEAM', avatar: '/img/authors/research.svg' },
    date: '2025-11-20',
    image: '/img/blog/default-post.svg',
  },
];

const CATEGORIES: Category[] = ['All', 'Product', 'For Business', 'Developers', 'Research', 'Company'];

export default function BlogListing(): JSX.Element {
  const [activeCategory, setActiveCategory] = useState<Category>('All');
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Get featured post (first one marked as featured)
  const featuredPost = useMemo(() => BLOG_POSTS.find((p) => p.featured), []);

  // Filter posts by category and search query (excluding featured from main grid)
  const filteredPosts = useMemo(() => {
    let posts = BLOG_POSTS.filter((p) => !p.featured);

    // Filter by category
    if (activeCategory !== 'All') {
      posts = posts.filter((p) => p.categories.includes(activeCategory));
    }

    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      posts = posts.filter((p) =>
        p.title.toLowerCase().includes(query) ||
        p.description?.toLowerCase().includes(query) ||
        p.author.name.toLowerCase().includes(query)
      );
    }

    return posts;
  }, [activeCategory, searchQuery]);

  // Check if featured post matches search query
  const showFeaturedPost = useMemo(() => {
    if (!featuredPost || activeCategory !== 'All') return false;
    if (!searchQuery.trim()) return true;

    const query = searchQuery.toLowerCase();
    return (
      featuredPost.title.toLowerCase().includes(query) ||
      featuredPost.description?.toLowerCase().includes(query) ||
      featuredPost.author.name.toLowerCase().includes(query)
    );
  }, [featuredPost, activeCategory, searchQuery]);

  return (
    <div className="blog-listing">
      {/* Page Header */}
      <header className="blog-listing__header">
        <h1 className="blog-listing__title">The Open Telco Blog</h1>
        <BlogSearch value={searchQuery} onChange={setSearchQuery} />
      </header>

      {/* Category Filter */}
      <CategoryFilter
        categories={CATEGORIES}
        activeCategory={activeCategory}
        onCategoryChange={setActiveCategory}
      />

      {/* Featured Post (Hero) */}
      {showFeaturedPost && featuredPost && (
        <BlogHero post={featuredPost} />
      )}

      {/* Blog Grid */}
      <section className="blog-listing__grid-section">
        {filteredPosts.length > 0 ? (
          <div className="blog-grid">
            {filteredPosts.map((post) => (
              <BlogCard key={post.slug} post={post} />
            ))}
          </div>
        ) : (
          <div className="blog-listing__empty">
            <p>No posts found{searchQuery ? ` matching "${searchQuery}"` : ' in this category'}.</p>
          </div>
        )}
      </section>
    </div>
  );
}
