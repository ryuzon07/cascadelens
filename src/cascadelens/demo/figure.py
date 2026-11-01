"""The static graph figure: suspects coloured by score, with the abstain banner and disclaimer."""

from pathlib import Path

import matplotlib as mpl
import networkx as nx  # type: ignore[import-untyped]
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from cascadelens.eval.plots import DPI, GRID, INK, MUTED_INK, SURFACE
from cascadelens.inference import CaseInput, DiagnoseResult

BANNER = "ABSTAIN: low confidence"
NO_TOPOLOGY = "No topology for this case: nodes are shown without edges"
UNRANKED = "#d9d9d6"
CRITICAL = "#e34948"
RAMP = LinearSegmentedColormap.from_list("suspects", ["#d6e6f8", "#2a78d6", "#0d3a75"])
LABELLED = 3
LAYOUT_SEED = 0
NODE_SIZE = 620


def case_graph(case: CaseInput, result: DiagnoseResult) -> nx.DiGraph:
    """Return every known service as a node, with the recorded calls as edges."""
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(set(case.metrics["service"].to_list())))
    graph.add_nodes_from(item.service for item in result.ranking)
    for edge in case.topology or ():
        graph.add_edge(edge.caller, edge.callee)
    return graph


def _positions(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    if graph.number_of_edges():
        placed = nx.spring_layout(graph, seed=LAYOUT_SEED)
    else:
        placed = nx.circular_layout(sorted(graph.nodes))
    return {node: (float(point[0]), float(point[1])) for node, point in placed.items()}


def _draw(axes: Axes, case: CaseInput, result: DiagnoseResult) -> Normalize:
    graph = case_graph(case, result)
    places = _positions(graph)
    scores = {item.service: item.score for item in result.ranking}
    scale = Normalize(min(scores.values()), max(scores.values()))
    colors = [RAMP(scale(scores[node])) if node in scores else UNRANKED for node in graph.nodes]
    if graph.number_of_edges():
        nx.draw_networkx_edges(
            graph, places, ax=axes, edge_color=GRID, node_size=NODE_SIZE, arrowsize=12, width=1.2
        )
    nx.draw_networkx_nodes(
        graph,
        places,
        ax=axes,
        node_color=colors,
        node_size=NODE_SIZE,
        edgecolors=SURFACE,
        linewidths=2,
    )
    for rank, item in enumerate(result.ranking[:LABELLED], start=1):
        x, y = places[item.service]
        axes.annotate(
            f"{rank}. {item.service}",
            (x, y),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": SURFACE, "edgecolor": GRID},
        )
    axes.margins(0.22)
    axes.set_axis_off()
    return scale


def render_figure(case: CaseInput, result: DiagnoseResult) -> Figure:
    """Return the figure of a diagnosis, without a display."""
    figure = Figure(figsize=(8.0, 5.6), facecolor=SURFACE, layout="constrained")
    axes = figure.add_subplot()
    axes.set_facecolor(SURFACE)
    axes.set_title(
        f"Likely origin services, {case.case_id}, confidence {result.confidence:.2f}"
        + ("" if result.calibrated else " (uncalibrated)"),
        color=INK,
        loc="left",
        fontsize=11,
    )
    scale = _draw(axes, case, result)
    bar = figure.colorbar(ScalarMappable(norm=scale, cmap=RAMP), ax=axes, fraction=0.035, ticks=[])
    bar.set_label("Score, low to high. Gray: not ranked", color=MUTED_INK, fontsize=9)
    bar.outline.set_visible(False)
    if result.abstain:
        axes.text(
            0.5,
            0.98,
            f"! {BANNER}",
            transform=axes.transAxes,
            ha="center",
            va="top",
            color=INK,
            fontsize=12,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": SURFACE, "edgecolor": CRITICAL},
        )
    if case.topology is None:
        axes.text(
            0.5,
            0.02,
            NO_TOPOLOGY,
            transform=axes.transAxes,
            ha="center",
            color=MUTED_INK,
            fontsize=9,
        )
    figure.supxlabel(result.disclaimer, color=MUTED_INK, fontsize=9)
    return figure


def save_figure(case: CaseInput, result: DiagnoseResult, out: Path) -> Path:
    """Write the figure as a PNG."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"svg.hashsalt": "cascadelens"}):
        render_figure(case, result).savefig(out, dpi=DPI, metadata={"Software": None})
    return out
