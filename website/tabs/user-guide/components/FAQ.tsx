import React from 'react';
import { useAccordion } from '../../../src/hooks/useAccordion';

interface FAQItem {
  question: string;
  answer: React.ReactNode;
}

interface FAQProps {
  items: FAQItem[];
  title?: string;
}

export default function FAQ({ items, title }: FAQProps): JSX.Element {
  const { isOpen, handleToggle } = useAccordion();

  return (
    <div className="faq-section">
      {title && <h3 className="faq-title">{title}</h3>}
      <div className="faq-accordion">
        {items.map((item, index) => (
          <div
            key={index}
            className={`faq-item ${isOpen(index) ? 'open' : ''}`}
          >
            <button
              className="faq-question"
              onClick={() => handleToggle(index)}
              aria-expanded={isOpen(index)}
            >
              <span>{item.question}</span>
              <span className="faq-icon">
                {isOpen(index) ? '−' : '+'}
              </span>
            </button>
            <div className="faq-answer">
              <div className="faq-answer-content">
                {item.answer}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export type { FAQItem };
