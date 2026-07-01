from langgraph_harness.routing import (
    route_after_generate,
    route_after_verify,
)


def state(**updates):
    value = {
        "errors": [],
        "verify_result": {},
        "iteration": 1,
        "max_iterations": 3,
    }
    value.update(updates)
    return value


def test_route_after_generate_stops_on_error() -> None:
    assert route_after_generate(state(errors=["boom"])) == "error"


def test_route_after_verify_success_retry_and_limit() -> None:
    assert route_after_verify(state(verify_result={"passed": True})) == "success"
    assert route_after_verify(state()) == "retry"
    assert (
        route_after_verify(state(iteration=3, max_iterations=3))
        == "max_iter_reached"
    )
