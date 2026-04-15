from __future__ import annotations

from django.urls import path

from gatewayai.server import passthrough, views

llm_routes = [
    path(
        "llm/stream/",
        views.stream_completion,
        name="llm_stream",
        kwargs=dict(methods=["POST"]),
    ),
    path(
        "llm/complete/",
        views.complete,
        name="llm_complete",
        kwargs=dict(methods=["POST"]),
    ),
    path(
        "llm/models/",
        views.list_models,
        name="llm_models",
        kwargs=dict(methods=["GET"]),
    ),
    path(
        "llm/<str:provider>/<path:path>",
        passthrough.passthrough,
        name="llm_passthrough",
    ),
]


def get_urlpatterns():
    return llm_routes
