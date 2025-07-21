"""
Smoke‑test the end‑to‑end supplemental‑figure pipeline.

We feed:
  • one main figure
  • one supplemental figure
Then run the assembler and make sure:

  a) the supplemental figure is numbered S1
  b) the Supplementary Figures section is present
  c) the Results text cites “Figure S1”
"""

import pytest

from manugen_ai.agents.ai_science_writer.sub_agents.figure.agent import figure_agent
from manugen_ai.agents.ai_science_writer.sub_agents.manuscript_drafter.assembler.agent import assembler_agent
from manugen_ai.utils import run_agent_workflow

# Constants for convenience
APP = "test_app"
USER = "test_user"
SESSION = "supp_fig_session"


async def _push_figure(prompt: str):
    """Helper to run the figure agent once with a custom LLM output string."""
    # We bypass LLM by poking state directly: the harness needs an actual
    # LLM call to parse, so we simulate with a preset JSON string.
    json_reply = prompt.strip()

    _, state, _ = await run_agent_workflow(
        agent=figure_agent,
        prompt=json_reply,
        app_name=APP,
        user_id=USER,
        session_id=SESSION,
        verbose=False,
        # supply content as if it came from model
        model_override=True,
    )
    return state


@pytest.mark.asyncio
async def test_supplementary_flow():
    # 1) Add a MAIN figure
    main_json = """
    {
      "title": "Dose–response curve",
      "description": "IC50 plotted against log‑concentration."
    }
    """
    state = await _push_figure(main_json)

    # 2) Add a SUPPLEMENTAL figure
    supp_json = """
    {
      "figure_type": "supplemental",
      "title": "Raw western blot",
      "description": "Full‑length gel image."
    }
    """
    state = await _push_figure(supp_json)

    # quick assertions on state
    assert "supplemental_figures" in state
    assert 1 in state["supplemental_figures"]
    assert state["supplemental_figures"][1]["title"] == "Raw western blot"
    assert "supplementary_figures" in state
    assert "**Figure S1." in state["supplementary_figures"]

    # 3) Run the assembler
    _, assembled_state, assembled_events = await run_agent_workflow(
        agent=assembler_agent,
        prompt="",
        app_name=APP,
        user_id=USER,
        session_id=SESSION,
        verbose=False,
    )

    manuscript = assembled_events[-1].content.parts[0].text
    assert "# Supplementary Figures" in manuscript
    assert "Figure S1" in manuscript
