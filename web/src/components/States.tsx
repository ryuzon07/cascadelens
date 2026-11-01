import type { ReactNode } from "react";
import { Button } from "./Button";
import styles from "./primitives.module.css";

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className={styles.state}>
      <p className={styles.stateTitle}>{title}</p>
      {children && <p className={styles.stateText}>{children}</p>}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message = error instanceof Error ? error.message : "Something went wrong.";
  return (
    <div className={styles.state} role="alert">
      <p className={styles.stateTitle}>This could not be loaded</p>
      <p className={styles.stateText}>{message}</p>
      {onRetry && <Button onClick={onRetry}>Retry</Button>}
    </div>
  );
}

/** Placeholder rows that hold a screen's shape while data loads. */
export function Skeleton({ rows = 3, label = "Loading" }: { rows?: number; label?: string }) {
  return (
    <div role="status" aria-label={label}>
      {Array.from({ length: rows }, (_, index) => (
        <span key={index} className={styles.skeleton} />
      ))}
    </div>
  );
}

export function StaleIndicator({ active }: { active: boolean }) {
  return (
    <span role="status" className={styles.stale}>
      {active ? "Refreshing…" : ""}
    </span>
  );
}
