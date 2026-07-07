from langgraph.graph import END, START, StateGraph

from langgraph_harness.nodes import GEMSNodes, NodeDependencies
from langgraph_harness.routing import (
    route_after_generate,
    route_after_planning,
    route_after_verify,
)
from langgraph_harness.states import AgentState


def build_graph(dependencies: NodeDependencies):
    nodes = GEMSNodes(dependencies)
    graph = StateGraph(AgentState)

    graph.add_node("skill_router", nodes.skill_router)
    graph.add_node("planner", nodes.planner)
    graph.add_node("decomposer", nodes.decomposer)
    graph.add_node("generator", nodes.generator)
    graph.add_node("verifier", nodes.verifier)
    graph.add_node("ocr_verifier", nodes.ocr_verifier)
    graph.add_node("memory_writer", nodes.memory_writer)
    graph.add_node("refiner", nodes.refiner)
    graph.add_node("finalizer", nodes.finalizer)

    graph.add_edge(START, "skill_router")
    graph.add_edge("skill_router", "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planning,
        {"continue": "decomposer", "error": "finalizer"},
    )
    graph.add_conditional_edges(
        "decomposer",
        route_after_planning,
        {"continue": "generator", "error": "finalizer"},
    )
    graph.add_conditional_edges(
        "generator",
        route_after_generate,
        {"verify": "verifier", "error": "finalizer"},
    )
    graph.add_edge("verifier", "ocr_verifier")
    graph.add_edge("ocr_verifier", "memory_writer")
    graph.add_conditional_edges(
        "memory_writer",
        route_after_verify,
        {
            "success": "finalizer",
            "retry": "refiner",
            "max_iter_reached": "finalizer",
            "error": "finalizer",
        },
    )
    graph.add_conditional_edges(
        "refiner",
        route_after_planning,
        {"continue": "generator", "error": "finalizer"},
    )
    graph.add_edge("finalizer", END)
    return graph.compile()
