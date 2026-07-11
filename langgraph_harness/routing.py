from typing import Literal

from langgraph_harness.states import AgentState


def route_after_planning(
    state: AgentState,
) -> Literal["continue", "error"]:
    return "error" if state["errors"] else "continue"


def route_after_generate(
    state: AgentState,
) -> Literal["verify", "error"]:
    return "error" if state["errors"] else "verify"


def route_after_verify(
    state: AgentState,
) -> Literal["success", "retry", "local_edit", "max_iter_reached", "error"]:
    if state["errors"]:
        return "error"
    if state["verify_result"].get("passed") is True:
        return "success"
    if state.get("recommended_action") == "local_edit" and state.get(
        "pending_edit"
    ):
        return "local_edit"
    if state["iteration"] >= state["max_iterations"]:
        return "max_iter_reached"
    return "retry"
