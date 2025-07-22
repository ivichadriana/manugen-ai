"""
Unit‑tests for the new supplemental‑figure support.

Run locally with:
    pytest -q packages/manugen-ai/tests/test_supplementary_figures.py
"""

from types import SimpleNamespace

import pytest

from manugen_ai.schema import (
    CURRENT_FIGURE_KEY,
    FIGURES_KEY,
    SUPP_FIGURES_KEY,
    SingleFigureDescription,
)
from manugen_ai.agents.ai_science_writer.sub_agents.figure.agent import (
    update_figure_state,
)

# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #


class DummyCtx(SimpleNamespace):
    """Minimal substitute for google.adk.agents.callback_context.CallbackContext."""

    def __init__(self, state):
        super().__init__(state=state)


def _make_current_fig(number: int, fig_type: str = "supplemental") -> dict:
    """
    Build the `CURRENT_FIGURE_KEY` payload expected by `update_figure_state`.

    Parameters
    ----------
    number : int
        The figure number to insert.
    fig_type : {'main', 'supplemental'}
        Which bucket the figure belongs to.
    """
    return {
        "figure_number": number,
        "figure_type": fig_type,
        "title": f"Example {fig_type.title()} Figure {number}",
        "description": "Lorem ipsum dolor sit amet.",
    }


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_single_supp_fig_updates_state():
    """
    One supplementary figure should:
      * be placed in `SUPP_FIGURES_KEY`
      * clear the CURRENT_FIGURE_KEY
      * prepend a caption to the running `state['supplementary_figures']` string
    """
    state = {CURRENT_FIGURE_KEY: _make_current_fig(1, "supplemental")}
    ctx = DummyCtx(state)

    # Act
    update_figure_state(ctx)

    # Assert
    assert CURRENT_FIGURE_KEY in state and state[CURRENT_FIGURE_KEY] == ""
    assert SUPP_FIGURES_KEY in state
    assert state[SUPP_FIGURES_KEY] == {
        1: {
            "figure_type": "supplemental",
            "title": "Example Supplemental Figure 1",
            "description": "Lorem ipsum dolor sit amet.",
        }
    }
    assert "**Figure S1." in state.get("supplementary_figures", "")


def test_multiple_supp_figs_increment_numbers():
    """
    Adding two supplementary figures sequentially should create
    S1 and S2 entries and append captions for both.
    """
    state = {}

    # First supplementary figure
    state[CURRENT_FIGURE_KEY] = _make_current_fig(1, "supplemental")
    update_figure_state(DummyCtx(state))

    # Second supplementary figure
    state[CURRENT_FIGURE_KEY] = _make_current_fig(2, "supplemental")
    update_figure_state(DummyCtx(state))

    supp_bucket = state[SUPP_FIGURES_KEY]
    assert set(supp_bucket.keys()) == {1, 2}
    assert "**Figure S1." in state["supplementary_figures"]
    assert "**Figure S2." in state["supplementary_figures"]


def test_main_and_supp_buckets_are_independent():
    """
    Main‑figures should live in `FIGURES_KEY`; supplementary ones in
    `SUPP_FIGURES_KEY`, with independent numbering.
    """
    state = {}

    # Add main figure 1
    state[CURRENT_FIGURE_KEY] = _make_current_fig(1, "main")
    update_figure_state(DummyCtx(state))

    # Add supplementary figure 1 (S1)
    state[CURRENT_FIGURE_KEY] = _make_current_fig(1, "supplemental")
    update_figure_state(DummyCtx(state))

    assert FIGURES_KEY in state and SUPP_FIGURES_KEY in state
    assert set(state[FIGURES_KEY].keys()) == {1}
    assert set(state[SUPP_FIGURES_KEY].keys()) == {1}
    # Captions should reference S1 only once
    assert state["supplementary_figures"].count("**Figure S1.") == 1
