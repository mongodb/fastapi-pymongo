"""Unit tests for the optional Beanie lifespan wiring.

Skipped unless the ``beanie`` extra is installed. ``init_beanie`` is
monkeypatched, so no MongoDB deployment is required.
"""

from importlib.metadata import version

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pymongo import AsyncMongoClient

from fastapi_pymongo import MongoSettings
from fastapi_pymongo.lifespan import _STATE_KEY

URI = "mongodb://localhost:27017"


def test_beanie_lifespan_runs_init_and_closes_client(monkeypatch):
    beanie = pytest.importorskip("fastapi_pymongo.beanie")

    calls = []

    async def fake_init_beanie(*, database, document_models):
        calls.append((database, document_models))

    monkeypatch.setattr(beanie, "init_beanie", fake_init_beanie)

    settings = MongoSettings(uri=URI, database="myapp")
    app = FastAPI(lifespan=beanie.beanie_lifespan(settings, document_models=[]))

    with TestClient(app):
        client = getattr(app.state, _STATE_KEY)
        assert isinstance(client, AsyncMongoClient)
        assert client._closed is False
        driver = client.options._options["driver"]
        assert driver.name == "fastapi-pymongo[beanie]"
        assert driver.version == version("fastapi-pymongo")

    assert client._closed is True
    assert len(calls) == 1
    database, document_models = calls[0]
    assert database.name == "myapp"
    assert document_models == []
