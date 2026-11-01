import type { ReactNode } from "react";
import styles from "./primitives.module.css";

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return <section className={`${styles.card} ${className ?? ""}`}>{children}</section>;
}

export function CardHeader({ title, actions }: { title: string; actions?: ReactNode }) {
  return (
    <header className={styles.cardHeader}>
      <h2 className={styles.caption}>{title}</h2>
      {actions}
    </header>
  );
}

export function CardBody({ children }: { children: ReactNode }) {
  return <div className={styles.cardBody}>{children}</div>;
}

export const PENDING = "[pending]";

/** A headline number. A missing value shows as pending, never as a made-up number. */
export function KpiTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: string | null;
  hint?: string;
}) {
  return (
    <Card className={styles.tile}>
      <h2 className={styles.caption}>{label}</h2>
      <p className={styles.tileValue}>{value ?? PENDING}</p>
      {hint && <p className={styles.tileHint}>{hint}</p>}
    </Card>
  );
}
