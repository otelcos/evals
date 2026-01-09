import React from 'react';

export type Category = 'All' | 'Product' | 'For Business' | 'Developers' | 'Research' | 'Company';

interface CategoryFilterProps {
  categories: Category[];
  activeCategory: Category;
  onCategoryChange: (category: Category) => void;
}

export default function CategoryFilter({
  categories,
  activeCategory,
  onCategoryChange,
}: CategoryFilterProps): JSX.Element {
  return (
    <div className="category-tabs">
      {categories.map((category) => (
        <button
          key={category}
          className={`category-tab ${
            category === 'All' ? 'category-tab--all' : ''
          } ${
            activeCategory === category ? 'category-tab--active' : ''
          }`}
          onClick={() => onCategoryChange(category)}
          aria-pressed={activeCategory === category}
        >
          {category}
        </button>
      ))}
    </div>
  );
}
