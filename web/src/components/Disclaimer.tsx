import { Tag } from "./Tag";

export const DISCLAIMER = "Ranking of likely origin services, not proof of causation.";
export const READ_ONLY = "CascadeLens is read-only: it changes nothing in your systems.";

export function Disclaimer() {
  return <p>{DISCLAIMER}</p>;
}

export function SampleDataBadge() {
  return <Tag value="Sample data" tone="warn" />;
}
