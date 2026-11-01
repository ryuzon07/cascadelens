import type { UseQueryResult } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { ErrorState, Skeleton, StaleIndicator } from "./States";

interface QueryViewProps<T> {
  query: UseQueryResult<T>;
  label: string;
  rows?: number;
  children: (data: T) => ReactNode;
}

/** The loading, error and refetching states every screen needs. */
export function QueryView<T>({ query, label, rows, children }: QueryViewProps<T>) {
  if (query.isPending) return <Skeleton label={`Loading ${label}`} rows={rows} />;
  if (query.isError) return <ErrorState error={query.error} onRetry={() => void query.refetch()} />;
  return (
    <>
      <StaleIndicator active={query.isFetching} />
      {children(query.data)}
    </>
  );
}
