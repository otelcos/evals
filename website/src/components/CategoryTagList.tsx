import React from 'react';
import { getCategoryColors } from '../../tabs/blog/constants/categoryColors';

interface CategoryTagListProps {
  categories: string[];
  className?: string;
}

export default function CategoryTagList({
  categories,
  className = 'blog-card__categories',
}: CategoryTagListProps): JSX.Element {
  return (
    <div className={className}>
      {categories.map((category) => {
        const colors = getCategoryColors(category);
        return (
          <span
            key={category}
            className="category-tag"
            style={{
              color: colors.text,
              borderColor: colors.border,
            }}
          >
            {category}
          </span>
        );
      })}
    </div>
  );
}
