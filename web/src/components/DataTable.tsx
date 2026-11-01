import { ArrowDown, ArrowUp } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import styles from "./primitives.module.css";

export interface Column<Row> {
  key: string;
  header: string;
  render: (row: Row) => ReactNode;
  /** A column with a sort value gets a sortable header. */
  sortValue?: (row: Row) => string | number;
}

interface DataTableProps<Row> {
  caption: string;
  columns: Column<Row>[];
  rows: Row[];
  rowKey: (row: Row) => string;
}

type Sort = { key: string; descending: boolean } | null;

export function DataTable<Row>({ caption, columns, rows, rowKey }: DataTableProps<Row>) {
  const [sort, setSort] = useState<Sort>(null);

  const ordered = useMemo(() => {
    const column = columns.find((item) => item.key === sort?.key);
    const value = column?.sortValue;
    if (!value) return rows;
    const direction = sort?.descending ? -1 : 1;
    return [...rows].sort((a, b) => {
      const [first, second] = [value(a), value(b)];
      return first < second ? -direction : first > second ? direction : 0;
    });
  }, [columns, rows, sort]);

  function toggle(key: string) {
    setSort((current) =>
      current?.key === key ? { key, descending: !current.descending } : { key, descending: false },
    );
  }

  function direction(key: string): "ascending" | "descending" | undefined {
    if (sort?.key !== key) return undefined;
    return sort.descending ? "descending" : "ascending";
  }

  return (
    // The box scrolls on narrow screens, so keyboard users must be able to focus it.
    // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
    <div className={styles.tableBox} tabIndex={0} role="region" aria-label={caption}>
      <table className={styles.table}>
        <caption className="visually-hidden">{caption}</caption>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} scope="col" aria-sort={direction(column.key)}>
                {column.sortValue ? (
                  <button type="button" className={styles.sort} onClick={() => toggle(column.key)}>
                    {column.header}
                    {direction(column.key) === "ascending" && (
                      <ArrowUp size={12} aria-hidden="true" />
                    )}
                    {direction(column.key) === "descending" && (
                      <ArrowDown size={12} aria-hidden="true" />
                    )}
                  </button>
                ) : (
                  column.header
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ordered.map((row) => (
            <tr key={rowKey(row)}>
              {columns.map((column) => (
                <td key={column.key}>{column.render(row)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
