import React from 'react';
import { BlogPost } from './BlogCard';
import { useBlogPostUrls } from '../../../src/hooks/useBlogPostUrls';
import { formatDate } from '../utils/formatDate';
import CategoryTagList from '../../../src/components/CategoryTagList';

interface BlogHeroProps {
  post: BlogPost;
}

export default function BlogHero({ post }: BlogHeroProps): JSX.Element {
  const { postUrl, imageUrl, avatarUrl } = useBlogPostUrls({
    slug: post.slug,
    image: post.image,
    authorAvatar: post.author.avatar,
  });

  return (
    <section className="featured-post">
      {/* Image - Left Side */}
      <a
        href={postUrl}
        className="featured-post__image"
        aria-label={`Read ${post.title}`}
      >
        <img
          src={imageUrl}
          alt=""
          width={640}
          height={400}
        />
      </a>

      {/* Content - Right Side */}
      <div className="featured-post__content">
        {/* Author + Date */}
        <div className="featured-post__author">
          <img
            src={avatarUrl}
            alt=""
            className="featured-post__avatar"
            width={24}
            height={24}
          />
          <span className="featured-post__meta">
            {post.author.name} — {formatDate(post.date)}
          </span>
        </div>

        {/* Title */}
        <h2 className="featured-post__title">
          <a href={postUrl}>{post.title}</a>
        </h2>

        {/* Category Tags */}
        <CategoryTagList categories={post.categories} className="featured-post__categories" />

        {/* Read More Link */}
        <a
          href={postUrl}
          className="featured-post__read-more"
          aria-label={`Read full article: ${post.title}`}
        >
          READ FULL ARTICLE
          <svg
            className="featured-post__read-more-icon"
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
      </div>
    </section>
  );
}
