import { useState, useCallback } from 'react';

interface UseAccordionResult {
  openIndex: number | null;
  handleToggle: (index: number) => void;
  isOpen: (index: number) => boolean;
}

export function useAccordion(initialIndex: number | null = null): UseAccordionResult {
  const [openIndex, setOpenIndex] = useState<number | null>(initialIndex);

  const handleToggle = useCallback((index: number) => {
    setOpenIndex(prev => (prev === index ? null : index));
  }, []);

  const isOpen = useCallback((index: number) => openIndex === index, [openIndex]);

  return { openIndex, handleToggle, isOpen };
}
