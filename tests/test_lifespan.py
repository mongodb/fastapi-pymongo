"""Unit tests for the client lifespan and its FastAPI dependencies.

No MongoDB deployment is required: ``AsyncMongoClient`` connects lazily, so
the lifespan can be started and stopped without touching a server.
"""

import asyncio
from importlib.metadata import version

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from fastapi_pymongo import MongoSettings, get_client, mongo_lifespan
from fastapi_pymongo.lifespan import _STATE_KEY, get_database

URI = "mongodb://localhost:27017"


@pytest.fixture
def client():
    """A client that is closed after the test, whether it passes or fails."""
    client = AsyncMongoClient(URI)
    yield client
    asyncio.run(client.close())


def _request_with_client(client: AsyncMongoClient) -> Request:
    app = FastAPI()
    setattr(app.state, _STATE_KEY, client)
    return Request({"type": "http", "app": app})


def test_mongo_lifespan_stores_client_with_driver_metadata():
    """The lifespan-created client carries the DriverInfo set at construction."""
    app = FastAPI(lifespan=mongo_lifespan(MongoSettings(uri=URI)))

    with TestClient(app):
        client = getattr(app.state, _STATE_KEY)
        assert isinstance(client, AsyncMongoClient)
        driver = client.options._options["driver"]
        assert driver.name == "fastapi-pymongo"
        assert driver.version == version("fastapi-pymongo")


def test_mongo_lifespan_closes_client_on_shutdown():
    app = FastAPI(lifespan=mongo_lifespan(MongoSettings(uri=URI)))

    with TestClient(app):
        client = getattr(app.state, _STATE_KEY)
        assert client._closed is False

    assert client._closed is True


def test_get_client_returns_state_client(client):
    assert get_client(_request_with_client(client)) is client


def test_get_client_without_lifespan_raises():
    with pytest.raises(RuntimeError, match="lifespan=mongo_lifespan"):
        get_client(Request({"type": "http", "app": FastAPI()}))


def test_get_database_returns_configured_database(client):
    settings = MongoSettings(uri=URI, database="myapp")

    database = get_database(settings)(_request_with_client(client))

    assert isinstance(database, AsyncDatabase)
    assert database.name == "myapp"
    assert database.client is client
