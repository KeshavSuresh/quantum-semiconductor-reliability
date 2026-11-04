"""Colour themes and shared layout for all figures.

The eight categorical colours are used in this fixed order, never cycled.
They were checked with a colour-vision-deficiency validator in both modes
(adjacent-pair CVD Delta E >= 8.4, normal-vision Delta E >= 19.3). Three light-mode
slots fall below 3:1 contrast against the surface, so every multi-series
figure also carries a legend and, for up to four series, direct end-of-line
labels, so identity never depends on colour alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go


@dataclass(frozen=True)
class Theme:
    """Colours for one display mode. Text uses ink colours, never series colours."""

    name: str
    surface: str
    ink_primary: str
    ink_secondary: str
    ink_muted: str
    gridline: str
    axis_line: str
    series: tuple[str, ...]

    @property
    def mixture(self) -> str:
        """Colour for the weighted mixture curve and exact reference: primary ink."""
        return self.ink_primary


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink_primary="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    gridline="#e1e0d9",
    axis_line="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink_primary="#f0efec",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    gridline="#2c2c2a",
    axis_line="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
)

#: Maximum number of series that get their own categorical colour.
MAX_CATEGORICAL = len(LIGHT.series)

LINE_WIDTH = 2
MIXTURE_LINE_WIDTH = 3
MARKER_SIZE = 9


def apply_layout(fig: go.Figure, theme: Theme, *, title: str, x_title: str, y_title: str) -> go.Figure:
    """Apply the shared, recessive chart chrome to ``fig`` in place and return it."""
    if not isinstance(theme, Theme):
        raise TypeError("theme must be a qsr.visualization.Theme (use LIGHT or DARK)")
    axis = dict(
        showgrid=True,
        gridcolor=theme.gridline,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=theme.axis_line,
        ticks="outside",
        tickcolor=theme.axis_line,
        tickfont=dict(color=theme.ink_muted),
        title_font=dict(color=theme.ink_secondary),
    )
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(color=theme.ink_primary, size=17)),
        paper_bgcolor=theme.surface,
        plot_bgcolor=theme.surface,
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", color=theme.ink_secondary, size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(color=theme.ink_secondary),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(bgcolor=theme.surface, bordercolor=theme.axis_line, font=dict(color=theme.ink_primary)),
        margin=dict(l=70, r=40, t=90, b=60),
    )
    fig.update_xaxes(title_text=x_title, **axis)
    fig.update_yaxes(title_text=y_title, **axis)
    return fig
