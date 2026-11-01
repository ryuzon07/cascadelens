import { AlertTriangle } from "lucide-react";
import type { ReactNode } from "react";
import type { IncidentDetail } from "../api/client";
import { formatPercent, formatUnit } from "../format";
import { PENDING } from "./Card";
import { ConfidenceGauge } from "./ConfidenceGauge";
import { Tag } from "./Tag";
import styles from "./decision.module.css";

interface DecisionCardProps {
  incident: IncidentDetail;
  telemetryCoverage: number | null;
  benchmarkTop3: number | null;
}

export function reasonFor(incident: IncidentDetail): string {
  const top = incident.diagnosis.ranking[0];
  if (!top) return "";
  const group = top.evidence.top_groups[0]?.group;
  const parts = group ? [`Largest contribution from ${group} features`] : [];
  if (top.evidence.onset_rank === 1) parts.push("among the earliest services to turn anomalous");
  return `${parts.join(", ")}.`;
}

export function abstainReasons(incident: IncidentDetail, coverage: number | null): string[] {
  const { ranking, confidence } = incident.diagnosis;
  const reasons: string[] = [];
  const threshold = incident.abstain_threshold;
  if (threshold !== null && threshold !== undefined) {
    reasons.push(
      `Confidence ${formatUnit(confidence)} is below the threshold ${formatUnit(threshold)}.`,
    );
  }
  const [first, second] = ranking;
  if (first && second) {
    const gap = (first.score - second.score).toFixed(2);
    reasons.push(`The top two leads are ${gap} apart in score.`);
  }
  if (coverage !== null && coverage < 1) {
    reasons.push(`Telemetry is incomplete: ${formatPercent(coverage)} of samples are present.`);
  }
  if (incident.topology_source === "inferred") {
    reasons.push("The call graph is inferred from metrics, not recorded.");
  }
  if (incident.topology_source === "none") reasons.push("No call graph is available.");
  return reasons;
}

function help(incident: IncidentDetail, coverage: number | null): string | null {
  const wanted = [];
  if (coverage !== null && coverage < 1) wanted.push("complete telemetry");
  if (incident.topology_source !== "provided") wanted.push("a recorded call graph");
  return wanted.length ? `What would help: ${wanted.join(" and ")}.` : null;
}

function Facts({ incident, telemetryCoverage, benchmarkTop3 }: DecisionCardProps): ReactNode {
  return (
    <dl className={styles.facts}>
      <div>
        <dt>Benchmark top-3</dt>
        <dd>{benchmarkTop3 === null ? PENDING : formatPercent(benchmarkTop3)}</dd>
      </div>
      <div>
        <dt>Telemetry coverage</dt>
        <dd>{telemetryCoverage === null ? PENDING : formatPercent(telemetryCoverage)}</dd>
      </div>
      <div>
        <dt>Latency</dt>
        <dd>{incident.diagnosis.timing.total_seconds.toFixed(2)} s</dd>
      </div>
    </dl>
  );
}

export function DecisionCard(props: DecisionCardProps) {
  const { incident, telemetryCoverage } = props;
  const threshold = incident.abstain_threshold ?? null;
  const gauge = (
    <ConfidenceGauge confidence={incident.diagnosis.confidence} threshold={threshold} />
  );
  if (incident.decision === "abstained") {
    const closing = help(incident, telemetryCoverage);
    return (
      <section className={styles.abstained} aria-label="Decision">
        <Tag value="abstained" />
        <h2 className={styles.headline}>
          <AlertTriangle size={16} aria-hidden="true" /> No confident answer. Investigate manually.
        </h2>
        <ul className={styles.reasons}>
          {abstainReasons(incident, telemetryCoverage).map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
          {closing && <li>{closing}</li>}
        </ul>
        {gauge}
        <Facts {...props} />
      </section>
    );
  }
  return (
    <section className={styles.confident} aria-label="Decision">
      <p className={styles.eyebrow}>Most likely origin</p>
      <p className={styles.service}>{incident.top_service}</p>
      {incident.decision === "uncalibrated" && (
        <p>
          <Tag value="uncalibrated" /> This model has no calibrator, so confidence is not a
          probability and it cannot abstain.
        </p>
      )}
      <p className={styles.reason}>{reasonFor(incident)}</p>
      {gauge}
      <Facts {...props} />
    </section>
  );
}
