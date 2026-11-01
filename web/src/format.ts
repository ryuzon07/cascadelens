/** A time in UTC, so that everyone reading an incident sees the same value. */
export function formatTime(seconds: number): string {
  return `${new Date(seconds * 1000).toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

/** A 0 to 1 value, with enough decimals that a value near 1 never rounds up to 1. */
export function formatUnit(value: number): string {
  const short = value.toFixed(3);
  if (Number(short) !== 1 || value === 1) return short;
  return value >= 0.999995 ? ">0.99999" : value.toFixed(5);
}

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatSigned(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}
