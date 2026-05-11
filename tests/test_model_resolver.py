"""Per-route model resolution.

The dispatcher and the worker MUST agree on which Ollama model a given
route uses. The resolver pins the precedence: explicit override > new
per-route mapping > legacy `large_model` / `default_local_model`.
"""
from main import _model_for_route
from inference.router import ROUTE_AI_LARGE, ROUTE_AI_SMALL, ROUTE_RULES


def _cfg(**settings):
    return {"settings": settings}


def test_explicit_override_always_wins():
    cfg = _cfg(default_local_model="default", models={"ai-small": "from-config"})
    assert _model_for_route(cfg, ROUTE_AI_SMALL, override="cli") == "cli"


def test_per_route_models_block_takes_priority_over_legacy_keys():
    cfg = _cfg(
        default_local_model="legacy-default",
        large_model="legacy-large",
        models={"ai-small": "small-7b", "ai-large": "large-14b"},
    )
    assert _model_for_route(cfg, ROUTE_AI_SMALL) == "small-7b"
    assert _model_for_route(cfg, ROUTE_AI_LARGE) == "large-14b"


def test_falls_back_to_large_model_for_ai_large_when_no_models_block():
    cfg = _cfg(default_local_model="default-7b", large_model="large-32b")
    assert _model_for_route(cfg, ROUTE_AI_LARGE) == "large-32b"


def test_falls_back_to_default_for_ai_small_when_no_models_block():
    cfg = _cfg(default_local_model="default-7b")
    assert _model_for_route(cfg, ROUTE_AI_SMALL) == "default-7b"


def test_falls_back_to_default_for_ai_large_when_large_model_missing():
    cfg = _cfg(default_local_model="default-7b")
    assert _model_for_route(cfg, ROUTE_AI_LARGE) == "default-7b"


def test_models_block_with_only_some_routes_populated():
    """Half-configured `models:` should still fall through to legacy keys."""
    cfg = _cfg(
        default_local_model="default-7b",
        large_model="legacy-large",
        models={"ai-small": "tiny-3b"},  # ai-large not specified
    )
    assert _model_for_route(cfg, ROUTE_AI_SMALL) == "tiny-3b"
    assert _model_for_route(cfg, ROUTE_AI_LARGE) == "legacy-large"


def test_rules_route_does_not_request_a_model():
    """Rules route should never ask the resolver — but if it does, fall
    back to the default rather than crashing."""
    cfg = _cfg(default_local_model="default-7b")
    assert _model_for_route(cfg, ROUTE_RULES) == "default-7b"
