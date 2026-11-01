import { useQuery } from "@tanstack/react-query";
import { Beaker, Boxes, FlaskConical, History, Network, Plug, Siren } from "lucide-react";
import { Link, NavLink } from "react-router-dom";
import { api, queryKeys } from "../api/client";
import { PENDING } from "./Card";
import { Select } from "./Fields";
import styles from "./shell.module.css";

const NAVIGATION = [
  { to: "/incidents", label: "Incidents", icon: Siren },
  { to: "/service-map", label: "Service map", icon: Network },
  { to: "/what-if", label: "What-if lab", icon: Beaker },
  { to: "/history", label: "History & feedback", icon: History },
  { to: "/model", label: "Model", icon: FlaskConical },
  { to: "/settings/integrations", label: "Integrations", icon: Plug },
];

function ModelHealth() {
  const { data, isError } = useQuery({
    queryKey: queryKeys.model,
    queryFn: ({ signal }) => api.model(signal),
  });
  const threshold = data?.abstain_threshold;
  const rows = [
    ["Version", data?.model_version],
    ["Calibrated", data && (data.calibrated ? "yes" : "no")],
    ["Threshold", threshold == null ? undefined : threshold.toFixed(2)],
    ["Frozen", data?.frozen_config_hash?.slice(0, 8) ?? (data ? "not frozen" : undefined)],
  ];
  return (
    <section className={styles.health} aria-label="Model health">
      <strong>Model</strong>
      {isError ? (
        <p>Not reachable</p>
      ) : (
        <dl>
          {rows.map(([name, value]) => (
            <div key={name} style={{ display: "contents" }}>
              <dt>{name}</dt>
              <dd>{value ?? PENDING}</dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}

export function Logo() {
  return <img src="/logo.svg" alt="" width={24} height={24} />;
}

export function Sidebar({ showComponents }: { showComponents: boolean }) {
  const items = showComponents
    ? [...NAVIGATION, { to: "/components", label: "Components", icon: Boxes }]
    : NAVIGATION;
  return (
    <aside className={styles.sidebar}>
      <Link to="/incidents" className={styles.brand}>
        <Logo />
        CascadeLens
      </Link>
      <Select
        label="Environment"
        options={[{ value: "local", label: "Local" }]}
        defaultValue="local"
      />
      <nav className={styles.nav} aria-label="Main">
        <ul>
          {items.map(({ to, label, icon: Icon }) => (
            <li key={to}>
              <NavLink to={to} className={styles.navLink}>
                <Icon size={15} aria-hidden="true" />
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      <ModelHealth />
      <p className={styles.user}>Signed in as local user</p>
    </aside>
  );
}
