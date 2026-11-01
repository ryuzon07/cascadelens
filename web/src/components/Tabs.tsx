import { useRef, type KeyboardEvent } from "react";
import styles from "./primitives.module.css";

export interface TabItem {
  id: string;
  label: string;
}

interface TabsProps {
  label: string;
  items: TabItem[];
  selected: string;
  onSelect: (id: string) => void;
}

/** Tabs with arrow-key movement, as the WAI-ARIA tabs pattern sets out. */
export function Tabs({ label, items, selected, onSelect }: TabsProps) {
  const list = useRef<HTMLDivElement>(null);

  function move(event: KeyboardEvent, index: number) {
    const steps: Record<string, number> = {
      ArrowRight: index + 1,
      ArrowLeft: index - 1 + items.length,
      Home: 0,
      End: items.length - 1,
    };
    const target = steps[event.key];
    if (target === undefined) return;
    event.preventDefault();
    const next = items[target % items.length];
    if (!next) return;
    onSelect(next.id);
    list.current?.querySelector<HTMLElement>(`[data-tab="${next.id}"]`)?.focus();
  }

  return (
    <div ref={list} role="tablist" aria-label={label} className={styles.tablist}>
      {items.map((item, index) => (
        <button
          key={item.id}
          type="button"
          role="tab"
          data-tab={item.id}
          aria-selected={item.id === selected}
          tabIndex={item.id === selected ? 0 : -1}
          className={styles.tab}
          onClick={() => onSelect(item.id)}
          onKeyDown={(event) => move(event, index)}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
