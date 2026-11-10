"""Unit tests for cursor pagination. These do not need a MongoDB server."""

import asyncio
import base64
from datetime import datetime
from uuid import uuid4

import pytest
from bson import BSON, Decimal128, ObjectId
from bson.binary import UuidRepresentation
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from pymongo import ASCENDING, DESCENDING

from fastapi_pymongo import PageParams, cursor_page, page_params
from fastapi_pymongo.pagination import (
    DEFAULT_PAGE_SIZE,
    InvalidCursor,
    _decode_cursor,
    _encode_cursor,
    _id_seek,
    _projection_returns_fully,
    _resolve_field,
    _seek_query,
)


class _StubCollection:
    """Just enough of a collection for validation-only paths.

    ``cursor_page`` reads ``codec_options`` before it queries, so tests that
    exercise cursor decoding (and nothing else) can pass this instead of a
    real collection.
    """

    codec_options = DEFAULT_CODEC_OPTIONS


@pytest.mark.parametrize(
    "sort_value",
    [
        42,
        "text",
        ObjectId(),
        # Naive datetimes round-trip exactly; BSON stores UTC milliseconds
        # and decodes to a naive datetime.
        datetime(2026, 10, 30, 12, 0, 0),
        Decimal128("3.14"),
        {"nested": [1, 2, 3]},
        None,
    ],
)
def test_cursor_round_trips_bson_values(sort_value):
    doc_id = ObjectId()
    token = _encode_cursor(sort_value, doc_id, ASCENDING)
    assert isinstance(token, str)
    decoded_value, decoded_id, decoded_direction = _decode_cursor(token)
    assert decoded_value == sort_value
    assert decoded_id == doc_id
    assert decoded_direction == ASCENDING


@pytest.mark.parametrize("direction", [ASCENDING, DESCENDING])
def test_cursor_round_trips_direction(direction):
    token = _encode_cursor("x", ObjectId(), direction)
    assert _decode_cursor(token)[2] == direction


def test_cursor_token_is_url_safe():
    token = _encode_cursor(datetime(2026, 1, 1, 12, 0, 0), ObjectId(), ASCENDING)
    assert all(char.isalnum() or char in "-_" for char in token)


def test_cursor_round_trips_uuid_with_codec_options():
    # uuidRepresentation=standard returns uuid.UUID values; the token must
    # round-trip them with the collection's codec options.
    options = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
    sort_value = uuid4()
    doc_id = uuid4()
    token = _encode_cursor(sort_value, doc_id, ASCENDING, options)
    assert _decode_cursor(token, options) == (sort_value, doc_id, ASCENDING)


def test_cursor_encode_rejects_uuid_without_representation():
    # The default codec options cannot encode a native UUID, which is why the
    # collection's options are threaded through.
    with pytest.raises(ValueError):
        _encode_cursor(uuid4(), ObjectId(), ASCENDING)


def test_decode_rejects_garbage():
    with pytest.raises(InvalidCursor):
        _decode_cursor("not a real token!!!")


def test_decode_rejects_base64_that_is_not_bson():
    token = base64.urlsafe_b64encode(b"plain text, not bson").rstrip(b"=").decode()
    with pytest.raises(InvalidCursor):
        _decode_cursor(token)


def test_decode_rejects_bson_missing_expected_keys():
    token = base64.urlsafe_b64encode(BSON.encode({"x": 1})).rstrip(b"=").decode()
    with pytest.raises(InvalidCursor):
        _decode_cursor(token)


def test_decode_rejects_token_without_direction():
    payload = BSON.encode({"sort_value": "x", "_id": ObjectId()})
    token = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    with pytest.raises(InvalidCursor):
        _decode_cursor(token)


def test_decode_rejects_empty_token():
    with pytest.raises(InvalidCursor):
        _decode_cursor("")


def test_invalid_cursor_is_a_400_http_exception():
    exc = InvalidCursor("bad token")
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 400
    assert exc.detail == "bad token"


def test_invalid_cursor_returns_400_without_a_handler():
    # A bad cursor raises while decoding, before the collection is queried, so
    # a stub carrying only codec options stands in for it.
    app = FastAPI()

    @app.get("/items")
    async def items(params: PageParams = Depends(page_params)):
        return await cursor_page(_StubCollection(), params)

    with TestClient(app) as client:
        response = client.get("/items", params={"cursor": "garbage"})
    assert response.status_code == 400
    # The detail is generic: the offending token must not be echoed back.
    assert response.json()["detail"] == "Invalid cursor"
    assert "garbage" not in response.text


@pytest.mark.parametrize("size", [0, -1])
def test_cursor_page_rejects_non_positive_size(size):
    # page_params bounds size at the HTTP layer; cursor_page enforces the
    # lower bound too, so a hand-built PageParams cannot silently return no
    # rows. The collection is unused: validation runs first.
    with pytest.raises(ValueError):
        asyncio.run(cursor_page(None, PageParams(size=size)))


@pytest.mark.parametrize("direction", [True, False, 1.0, 0, 2, "1"])
def test_cursor_page_rejects_non_direction_values(direction):
    # ``True == 1`` and ``1.0 == 1``, so a bool or float slips through a bare
    # ``in (ASCENDING, DESCENDING)`` check and fails deep in the driver. Reject
    # them like any other invalid direction. The collection is unused:
    # validation runs first.
    with pytest.raises(ValueError, match="direction must be"):
        asyncio.run(cursor_page(None, PageParams(size=2), direction=direction))


@pytest.mark.parametrize(
    "projection",
    [
        {"_id": 0, "n": 1},  # inclusion that drops _id
        {"extra": 1},  # inclusion that drops the sort field
        {"_id": 1},  # inclusion of _id only
        {"n": 0},  # exclusion of the sort field
        {"n": {"$slice": 1}},  # operator truncates the sort field
    ],
)
def test_cursor_page_rejects_projection_missing_cursor_fields(projection):
    # A projection that drops the sort field or _id would otherwise build a
    # silently wrong cursor (or raise KeyError on _id). Reject it up front;
    # the collection is unused because validation runs first.
    with pytest.raises(ValueError, match="projection must return"):
        asyncio.run(cursor_page(None, PageParams(size=2), sort_field="n", projection=projection))


@pytest.mark.parametrize(
    ("projection", "sort_field"),
    [
        ({"meta.rank": 1}, "meta"),  # inclusion of a child only
        ({"meta.rank": 0}, "meta"),  # exclusion of a child only
    ],
)
def test_cursor_page_rejects_partial_subdocument_sort_field(projection, sort_field):
    # A subdocument sort field must come back whole: a truncated subdoc makes
    # the next cursor seek re-match rows (duplicates), so reject it too.
    with pytest.raises(ValueError, match="projection must return"):
        asyncio.run(
            cursor_page(None, PageParams(size=2), sort_field=sort_field, projection=projection)
        )


def test_cursor_page_projection_message_dedupes_id_sort_field():
    # When the sort field is _id (the default), a dropped _id is one field,
    # not two, so it must not be named twice in the message.
    with pytest.raises(ValueError, match="projection must return _id in full"):
        asyncio.run(cursor_page(None, PageParams(size=2), projection={"_id": 0}))


@pytest.mark.parametrize(
    "projection",
    [
        {"n": 1},  # inclusion keeps sort field and _id by default
        {"extra": 0},  # exclusion of an unrelated field
        {"n": 1, "meta.rank": 1},  # dotted inclusion alongside the sort field
        {"_id": 1, "secret": 0},  # exclusion with an explicit _id: 1
    ],
)
def test_cursor_page_accepts_projection_that_returns_cursor_fields(projection):
    # Validation must not reject a projection that returns both fields whole.
    # The collection is still None: a valid projection falls through to the
    # query, so the AttributeError proves validation passed.
    with pytest.raises(AttributeError):
        asyncio.run(cursor_page(None, PageParams(size=2), sort_field="n", projection=projection))


# --- _projection_returns_fully: guards the projection/sort-field contract ---


@pytest.mark.parametrize(
    ("projection", "field", "expected"),
    [
        (None, "n", True),
        ({}, "n", True),
        ({"n": 1}, "n", True),
        ({"n": 1}, "_id", True),  # _id included by default
        ({"n": 1}, "other", False),
        ({"_id": 0, "n": 1}, "_id", False),
        ({"_id": 0, "n": 1}, "n", True),
        ({"_id": 1}, "_id", True),
        ({"_id": 1}, "n", False),
        ({"extra": 0}, "n", True),
        ({"n": 0}, "n", False),
        ({"_id": 0}, "_id", False),
        ({"meta": 1}, "meta.rank", True),  # projecting the parent keeps the child
        ({"meta.rank": 1}, "meta.rank", True),
        ({"meta.rank": 1}, "meta.other", False),
        ({"meta": 0}, "meta.rank", False),  # excluding the parent drops the child
        # Completeness: a subdocument field is only whole when it, or an
        # ancestor, is projected / when nothing overlapping it is excluded.
        ({"meta.rank": 1}, "meta", False),  # projecting a child truncates the parent
        ({"meta.rank": 0}, "meta", False),  # excluding a child truncates the parent
        # ``_id`` is legal in either form, so the other fields decide the mode:
        # a lone falsy field makes this an exclusion projection, not inclusion.
        ({"_id": 1, "secret": 0}, "n", True),
        ({"_id": 1, "secret": 0}, "secret", False),
        ({"_id": 1, "secret": 0}, "_id", True),
        # Operator dicts are truthy but return a truncated value, so they do
        # not keep the field whole.
        ({"n": {"$slice": 1}}, "n", False),
        ({"n": {"$elemMatch": {"x": 1}}}, "n", False),
        # A standalone $slice is an exclusion projection: unrelated fields,
        # including the sort field, still come back whole.
        ({"comments": {"$slice": 3}}, "n", True),
        ({"comments": {"$slice": 3}}, "comments", False),
        ({"comments": {"$slice": 3}, "n": 1}, "n", True),
        ({"comments": {"$slice": 3}, "n": 1}, "comments", False),
        # A string value is an expression that replaces the field, not a flag.
        ({"n": "$other"}, "n", False),
        ({"n": "1"}, "n", False),  # even a numeric string replaces the value
        ({"label": "0"}, "n", False),  # expression makes it inclusion, omitting n
        ({"label": "0"}, "label", False),
        ({"n": "1"}, "_id", True),  # _id still included by default
        # Non-boolean/number literals are computed inclusions that replace the
        # field, not flags.
        ({"n": []}, "n", False),
        ({"n": [1, 2]}, "n", False),
        ({"n": None}, "n", False),
        ({"n": datetime(2026, 1, 1)}, "n", False),
        ({"n": []}, "_id", True),  # still inclusion mode, _id kept
        # A truthy non-1 number is still an inclusion flag.
        ({"n": 2}, "n", True),
        # _id is truncated by a descendant projection or a computed value.
        ({"_id.part": 0}, "_id", False),
        ({"_id.part": 1}, "_id", False),
        ({"_id": {"$toString": "$_id"}}, "_id", False),
        # A sort field nested under _id is present when _id is included whole.
        ({"_id": 1}, "_id.rank", True),
        ({}, "_id.rank", True),  # everything is returned
        ({"n": 1}, "_id.rank", True),  # _id included by default
        ({"_id": 0}, "_id.rank", False),  # _id excluded
        ({"_id.rank": 1}, "_id.rank", True),
        ({"_id.rank.sub": 1}, "_id.rank", False),  # deeper path truncates it
        ({"_id.rank": 0}, "_id.rank", False),
    ],
)
def test_projection_returns_fully(projection, field, expected):
    assert _projection_returns_fully(projection, field) is expected


# --- _resolve_field: dotted-path lookup used to build the next cursor ---


@pytest.mark.parametrize(
    ("document", "path", "expected"),
    [
        ({"a": 1}, "a", 1),
        ({"a": {"b": 2}}, "a.b", 2),
        ({"a": {"b": {"c": 3}}}, "a.b.c", 3),
        ({"a": [{"b": 1}, {"b": 2}]}, "a.1.b", 2),
        ({"a": {"b": 1}}, "a.c", None),
        ({"a": 1}, "a.b", None),
        ({"a": [1]}, "a.5", None),
        ({}, "a.b", None),
    ],
)
def test_resolve_field(document, path, expected):
    assert _resolve_field(document, path) == expected


def test_resolve_field_rejects_array_traversal():
    # A named segment crossing an array is ordered by its minimum element, which
    # the seek cannot reproduce, so it is rejected.
    with pytest.raises(ValueError, match="traverses an array"):
        _resolve_field({"tags": [{"rank": 1}, {"rank": 2}]}, "tags.rank")


# --- _id_seek / _seek_query: the seek predicate, including nulls ---


def test_id_seek_uses_bson_total_order():
    # $expr compares across BSON types (a type-bracketed $gt/$lt would skip a
    # string id when seeking past an integer one), and $literal keeps a string
    # id from being read as a field path.
    oid = ObjectId()
    assert _id_seek(ASCENDING, oid) == {"$expr": {"$gt": ["$_id", {"$literal": oid}]}}
    assert _id_seek(DESCENDING, oid) == {"$expr": {"$lt": ["$_id", {"$literal": oid}]}}


def test_id_seek_null_id_uses_bson_order():
    # A null id is lowest in BSON order: first ascending, last descending.
    assert _id_seek(ASCENDING, None) == {"$expr": {"$gt": ["$_id", {"$literal": None}]}}
    assert _id_seek(DESCENDING, None) == {"$expr": {"$lt": ["$_id", {"$literal": None}]}}


def test_seek_query_wraps_sort_value_in_eq():
    # The sort value is client-supplied BSON; shorthand equality would run
    # ``{"$regex": ...}`` as a predicate, so it must be wrapped in $eq.
    oid = ObjectId()
    query = _seek_query({}, "n", ASCENDING, {"$regex": "x"}, oid)
    assert {
        "$and": [
            {"n": {"$eq": {"$regex": "x"}}},
            {"$expr": {"$gt": ["$_id", {"$literal": oid}]}},
        ]
    } in query["$or"]


def test_seek_query_null_id_tiebreaker_ascending():
    query = _seek_query({}, "n", ASCENDING, 5, None)
    assert {
        "$and": [
            {"n": {"$eq": 5}},
            {"$expr": {"$gt": ["$_id", {"$literal": None}]}},
        ]
    } in query["$or"]


def test_seek_query_default_id_sort_ascending_null_id():
    assert _seek_query({}, "_id", ASCENDING, None, None) == {
        "$expr": {"$gt": ["$_id", {"$literal": None}]}
    }


# --- page_params dependency, exercised through a real FastAPI app ---


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def handler(params: PageParams = Depends(page_params)):
        return {"cursor": params.cursor, "size": params.size}

    return app


def test_page_params_defaults():
    with TestClient(_app()) as client:
        assert client.get("/").json() == {"cursor": None, "size": DEFAULT_PAGE_SIZE}


def test_page_params_reads_query_string():
    with TestClient(_app()) as client:
        body = client.get("/", params={"cursor": "abc", "size": 5}).json()
    assert body == {"cursor": "abc", "size": 5}


@pytest.mark.parametrize("size", [0, -1])
def test_page_params_rejects_non_positive_size(size):
    with TestClient(_app()) as client:
        assert client.get("/", params={"size": size}).status_code == 422


def test_page_params_appear_in_openapi():
    parameters = {p["name"]: p for p in _app().openapi()["paths"]["/"]["get"]["parameters"]}
    assert set(parameters) == {"cursor", "size"}
    size_schema = parameters["size"]["schema"]
    assert size_schema["type"] == "integer"
    assert size_schema["default"] == DEFAULT_PAGE_SIZE
    assert size_schema["minimum"] == 1
    assert "maximum" not in size_schema
