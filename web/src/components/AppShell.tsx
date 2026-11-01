import { Outlet } from "react-router-dom";
import { showComponents } from "../config";
import { Disclaimer, READ_ONLY } from "./Disclaimer";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import styles from "./shell.module.css";

export function AppShell() {
  return (
    <div className={styles.shell}>
      <a href="#main" className={styles.skip}>
        Skip to content
      </a>
      <Sidebar showComponents={showComponents} />
      <div className={styles.column}>
        <TopBar />
        <main id="main" tabIndex={-1} className={styles.main}>
          <Outlet />
        </main>
        <footer className={styles.footer}>
          <Disclaimer />
          <p>{READ_ONLY}</p>
        </footer>
      </div>
    </div>
  );
}
