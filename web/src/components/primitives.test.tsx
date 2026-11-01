import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import { Button } from "./Button";
import { KpiTile, PENDING } from "./Card";
import { DataTable } from "./DataTable";
import { Select, Slider, TextInput, Toggle } from "./Fields";
import { EmptyState, ErrorState, Skeleton, StaleIndicator } from "./States";
import { Tabs } from "./Tabs";
import { Tag } from "./Tag";

describe("Button", () => {
  it("does not submit forms by default", () => {
    render(<Button>Run</Button>);
    expect(screen.getByRole("button", { name: "Run" })).toHaveAttribute("type", "button");
  });

  it("names an icon button", () => {
    render(
      <Button variant="icon" aria-label="Close">
        x
      </Button>,
    );
    expect(screen.getByRole("button", { name: "Close" })).toBeInTheDocument();
  });

  it("ignores clicks when disabled", async () => {
    const click = vi.fn();
    render(
      <Button disabled onClick={click}>
        Run
      </Button>,
    );
    await userEvent.click(screen.getByRole("button"));
    expect(click).not.toHaveBeenCalled();
  });
});

describe("Tag", () => {
  it.each([
    ["needs_feedback", "Needs feedback"],
    ["abstained", "Abstained"],
    ["top_1", "Top 1"],
  ])("states %s in words", (value, text) => {
    render(<Tag value={value} />);
    expect(screen.getByText(text)).toBeInTheDocument();
  });

  it("shows an unknown value as it is", () => {
    render(<Tag value="Sample data" tone="warn" />);
    expect(screen.getByText("Sample data")).toBeInTheDocument();
  });
});

describe("KpiTile", () => {
  it("shows a value", () => {
    render(<KpiTile label="Open" value="3" />);
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("shows pending for a missing value", () => {
    render(<KpiTile label="Open" value={null} />);
    expect(screen.getByText(PENDING)).toBeInTheDocument();
  });
});

function ControlledTabs() {
  const [selected, setSelected] = useState("a");
  return (
    <Tabs
      label="Views"
      selected={selected}
      onSelect={setSelected}
      items={[
        { id: "a", label: "A" },
        { id: "b", label: "B" },
        { id: "c", label: "C" },
      ]}
    />
  );
}

describe("Tabs", () => {
  it("moves with the arrow keys and wraps", async () => {
    render(<ControlledTabs />);
    await userEvent.tab();
    expect(screen.getByRole("tab", { name: "A" })).toHaveFocus();
    await userEvent.keyboard("{ArrowLeft}");
    expect(screen.getByRole("tab", { name: "C" })).toHaveAttribute("aria-selected", "true");
    await userEvent.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "A" })).toHaveFocus();
    await userEvent.keyboard("{End}");
    expect(screen.getByRole("tab", { name: "C" })).toHaveFocus();
    await userEvent.keyboard("{Home}");
    expect(screen.getByRole("tab", { name: "A" })).toHaveFocus();
  });

  it("keeps only the selected tab in the tab order", () => {
    render(<ControlledTabs />);
    expect(screen.getByRole("tab", { name: "B" })).toHaveAttribute("tabindex", "-1");
  });
});

describe("fields", () => {
  it("labels every control", () => {
    render(
      <>
        <TextInput label="Name" />
        <Select label="Kind" options={[{ value: "a", label: "A" }]} />
        <Slider label="Loss" value={30} min={0} max={100} step={10} onChange={() => {}} />
      </>,
    );
    expect(screen.getByLabelText("Name")).toBeInTheDocument();
    expect(screen.getByLabelText("Kind")).toBeInTheDocument();
    expect(screen.getByRole("slider")).toHaveAccessibleName(/Loss/);
  });

  it("gives a slider's value in words", () => {
    render(
      <Slider
        label="Loss"
        value={30}
        min={0}
        max={100}
        step={10}
        onChange={() => {}}
        format={(value) => `${value}%`}
      />,
    );
    expect(screen.getByRole("slider")).toHaveAttribute("aria-valuetext", "30%");
  });

  it("toggles and says which way it is set", async () => {
    const change = vi.fn();
    render(<Toggle label="Use graph" checked={false} onChange={change} />);
    const toggle = screen.getByRole("switch");
    expect(toggle).toHaveAttribute("aria-checked", "false");
    expect(toggle).toHaveTextContent("Off");
    await userEvent.click(toggle);
    expect(change).toHaveBeenCalledWith(true);
  });
});

describe("DataTable", () => {
  const rows = [
    { id: "b", size: 1 },
    { id: "a", size: 2 },
  ];
  const table = (
    <DataTable
      caption="Sizes"
      rows={rows}
      rowKey={(row) => row.id}
      columns={[
        { key: "id", header: "ID", render: (row) => row.id, sortValue: (row) => row.id },
        { key: "size", header: "Size", render: (row) => row.size },
      ]}
    />
  );

  function firstCell(): string | null {
    const body = screen.getAllByRole("rowgroup")[1];
    return within(body!).getAllByRole("cell")[0]!.textContent;
  }

  it("keeps the given order until sorted", () => {
    render(table);
    expect(firstCell()).toBe("b");
  });

  it("sorts both ways and announces it", async () => {
    render(table);
    const header = screen.getByRole("columnheader", { name: "ID" });
    await userEvent.click(screen.getByRole("button", { name: "ID" }));
    expect(firstCell()).toBe("a");
    expect(header).toHaveAttribute("aria-sort", "ascending");
    await userEvent.click(screen.getByRole("button", { name: "ID" }));
    expect(firstCell()).toBe("b");
    expect(header).toHaveAttribute("aria-sort", "descending");
  });

  it("leaves a column without a sort value as plain text", () => {
    render(table);
    expect(screen.queryByRole("button", { name: "Size" })).not.toBeInTheDocument();
  });

  it("can be reached by keyboard so it can scroll", () => {
    render(table);
    expect(screen.getByRole("region", { name: "Sizes" })).toHaveAttribute("tabindex", "0");
  });
});

describe("states", () => {
  it("shows an empty state", () => {
    render(<EmptyState title="Nothing here">Add one.</EmptyState>);
    expect(screen.getByText("Nothing here")).toBeInTheDocument();
  });

  it("shows an error with a retry", async () => {
    const retry = vi.fn();
    render(<ErrorState error={new ApiError(0, "network_error", "")} onRetry={retry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("could not be reached");
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(retry).toHaveBeenCalledOnce();
  });

  it("shows skeleton rows instead of a spinner", () => {
    render(<Skeleton rows={4} label="Loading incidents" />);
    expect(screen.getByRole("status", { name: "Loading incidents" }).children).toHaveLength(4);
  });

  it("announces a refetch", () => {
    render(<StaleIndicator active />);
    expect(screen.getByRole("status")).toHaveTextContent("Refreshing");
  });
});
