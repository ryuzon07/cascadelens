import { useQuery } from "@tanstack/react-query";
import { Bell } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { api, queryKeys } from "../api/client";
import { useMocks } from "../config";
import { Button } from "./Button";
import { SampleDataBadge } from "./Disclaimer";
import { TextInput } from "./Fields";
import styles from "./shell.module.css";

const NAMES: Record<string, string> = {
  incidents: "Incidents",
  "service-map": "Service map",
  "what-if": "What-if lab",
  history: "History & feedback",
  model: "Model",
  settings: "Settings",
  integrations: "Integrations",
  components: "Components",
};

export function Breadcrumbs() {
  const { pathname } = useLocation();
  const parts = pathname.split("/").filter(Boolean);
  return (
    <nav className={styles.crumbs} aria-label="Breadcrumb">
      <ol>
        {parts.map((part, index) => {
          const to = `/${parts.slice(0, index + 1).join("/")}`;
          const name = NAMES[part] ?? decodeURIComponent(part);
          const last = index === parts.length - 1;
          return (
            <li key={to}>
              {last ? <span aria-current="page">{name}</span> : <Link to={to}>{name}</Link>}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

/** Demo incidents come from the API, so the badge shows for seeded data as well as mocks. */
function useSampleData(): boolean {
  const query = { limit: 50 };
  const { data } = useQuery({
    queryKey: queryKeys.incidents(query),
    queryFn: ({ signal }) => api.incidents(query, signal),
  });
  return useMocks || Boolean(data?.items.some((item) => item.is_demo));
}

function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

function Notifications() {
  const [open, setOpen] = useState(false);
  return (
    <div className={styles.notifications}>
      <Button
        variant="icon"
        aria-label="Notifications"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        <Bell size={16} aria-hidden="true" />
      </Button>
      {open && <p className={styles.popover}>No notifications.</p>}
    </div>
  );
}

export function TopBar() {
  const search = useRef<HTMLInputElement>(null);
  const sample = useSampleData();

  useEffect(() => {
    function focusSearch(event: KeyboardEvent) {
      if (event.key !== "/" || event.ctrlKey || event.metaKey || event.altKey) return;
      if (isTyping(event.target)) return;
      event.preventDefault();
      search.current?.focus();
    }
    document.addEventListener("keydown", focusSearch);
    return () => document.removeEventListener("keydown", focusSearch);
  }, []);

  return (
    <header className={styles.topBar}>
      <Breadcrumbs />
      <div className={styles.search}>
        <TextInput ref={search} type="search" label="Search" hideLabel placeholder="Search  /" />
      </div>
      {sample && <SampleDataBadge />}
      <Notifications />
    </header>
  );
}
