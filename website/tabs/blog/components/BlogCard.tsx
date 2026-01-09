import React from 'react';
import { useBlogPostUrls } from '../../../src/hooks/useBlogPostUrls';
import { formatDate } from '../utils/formatDate';
import CategoryTagList from '../../../src/components/CategoryTagList';

export interface BlogPost {
  slug: string;
  title: string;
  description?: string;
  categories: string[];
  author: {
    name: string;
    avatar?: string;
  };
  date: string;
  image?: string;
  featured?: boolean;
}

interface BlogCardProps {
  post: BlogPost;
  variant?: 'default' | 'featured';
}

export default function BlogCard({ post, variant = 'default' }: BlogCardProps): JSX.Element {
  const { postUrl, imageUrl, avatarUrl } = useBlogPostUrls({
    slug: post.slug,
    image: post.image,
    authorAvatar: post.author.avatar,
  });

  return (
    <article className={`blog-card blog-card--${variant}`}>
      {/* Image */}
      <a href={postUrl} className="blog-card__image-link" aria-label={`Read ${post.title}`}>
        <div className="blog-card__image">
          <img
            src={imageUrl}
            alt=""
            loading="lazy"
            width={560}
            height={315}
          />
        </div>
      </a>

      {/* Author + Date (ABOVE title - Cohere style) */}
      <div className="blog-card__author">
        <img
          src={avatarUrl}
          alt=""
          className="blog-card__avatar"
          width={24}
          height={24}
        />
        <span className="blog-card__meta">
          {post.author.name} — {formatDate(post.date)}
        </span>
      </div>

      {/* Title */}
      <h3 className="blog-card__title">
        <a href={postUrl}>{post.title}</a>
      </h3>

      {/* Category Tags */}
      <CategoryTagList categories={post.categories} className="blog-card__categories" />

      {/* Read More Link */}
      <a
        href={postUrl}
        className="blog-card__read-more"
        aria-label={`Read full article: ${post.title}`}
      >
        READ FULL ARTICLE
        <svg
          className="blog-card__read-more-icon"
          viewBox="0 0 16 16"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
          <path
            d="M3 8H13M13 8L8.5 3.5M13 8L8.5 12.5"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </a>
    </article>
  );
}
