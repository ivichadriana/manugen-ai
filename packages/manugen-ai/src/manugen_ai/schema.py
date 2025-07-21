from __future__ import annotations

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from pydantic import BaseModel, Field


def prepare_instructions(callback_context: CallbackContext) -> Optional[types.Content]:
    current_state = callback_context.state.to_dict()

    for key1 in ManuscriptStructure.model_json_schema()["properties"].keys():
        # set instructions for each manuscript section
        if (
            INSTRUCTIONS_KEY in current_state
            and key1 in current_state[INSTRUCTIONS_KEY]
        ):
            callback_context.state[f"{INSTRUCTIONS_KEY}_{key1}"] = current_state[
                INSTRUCTIONS_KEY
            ][key1]

        # if there is no draft for this section, assign empty string
        if key1 not in callback_context.state:
            callback_context.state[key1] = ""

    # ----------------------------------------------------------------------
    # add figures descriptions (main + supplemental)
    figure_descriptions = ""

    # --- main figures ---
    if FIGURES_KEY in current_state:
        for num, fig in current_state[FIGURES_KEY].items():
            figure_descriptions += (
                f"Figure {num}: {fig['title']}\n{fig['description']}\n\n"
            )

    # --- supplemental figures ---
    if SUPP_FIGURES_KEY in current_state:
        for num, fig in current_state[SUPP_FIGURES_KEY].items():
            figure_descriptions += (
                f"Supplementary Figure S{num}: {fig['title']}\n{fig['description']}\n\n"
            )

    callback_context.state[FIGURES_DESCRIPTIONS_KEY] = figure_descriptions.strip()
    # ----------------------------------------------------------------------

class ManuscriptStructure(BaseModel):
    title: str = Field(default="")
    # keywords: str = Field(default="")
    abstract: str = Field(default="")
    introduction: str = Field(default="")
    results: str = Field(default="")
    discussion: str = Field(default="")
    methods: str = Field(default="")


INSTRUCTIONS_KEY = "instructions"
TITLE_KEY = "title"
ABSTRACT_KEY = "abstract"
INTRODUCTION_KEY = "introduction"
RESULTS_KEY = "results"
DISCUSSION_KEY = "discussion"
METHODS_KEY = "methods"


class SingleFigureDescription(BaseModel):
    figure_number: int = Field(default=0)
    title: str = Field(default="")
    description: str = Field(default="")
    figure_type: str = "main"   # allowed: "main" or "supplemental"


CURRENT_FIGURE_KEY = "current_figure"
FIGURES_KEY = "figures"
FIGURES_DESCRIPTIONS_KEY = "figures_descriptions"
SUPP_FIGURES_KEY = "supplemental_figures" 
SUPP_FIGURES_DESCRIPTIONS_KEY = "supplemental_figures_descriptions"
