"""Integration tests: round-trip through a real MongoDB deployment.

Requires a running server. Set MONGODB_URI (default mongodb://localhost:27017).
Skip these locally with: pytest -m "not integration".
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from bson import ObjectId
from bson.binary import UuidRepresentation
from bson.codec_options import CodecOptions
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field
from pymongo import ASCENDING, DESCENDING, AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from fastapi_pymongo import (
    InvalidCursor,
    MongoSettings,
    PageParams,
    PyObjectId,
    cursor_page,
    get_client,
    mongo_lifespan,
)
from fastapi_pymongo.lifespan import get_database
from fastapi_pymongo.pagination import _encode_cursor

pytestmark = pytest.mark.integration


class Item(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: PyObjectId = Field(alias="_id", default_factory=PyObjectId)
    name: str


@pytest.fixture
def settings():
    """Per-test settings pointing at a fresh, uniquely named database."""
    return MongoSettings(database=f"test_{uuid4().hex}")


@pytest.fixture
def app(settings):
    app = FastAPI(lifespan=mongo_lifespan(settings))
    db_dependency = get_database(settings)

    @app.post("/items", response_model=Item)
    async def create_item(item: Item, db: AsyncDatabase = Depends(db_dependency)):
        doc = item.model_dump(by_alias=True)
        await db.items.insert_one(doc)
        return doc

    @app.get("/items/{item_id}", response_model=Item)
    async def get_item(item_id: str, db: AsyncDatabase = Depends(db_dependency)):
        if not PyObjectId.is_valid(item_id):
            raise HTTPException(status_code=404, detail="Item not found")
        doc = await db.items.find_one({"_id": PyObjectId(item_id)})
        if doc is None:
            raise HTTPException(status_code=404, detail="Item not found")
        return doc

    return app


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def mongo_client():
    """An async client for direct database access."""
    async with AsyncMongoClient(MongoSettings().uri) as client:
        yield client


@pytest.fixture
async def db(mongo_client, settings):
    """The test database, dropped when the test finishes."""
    yield mongo_client[settings.database]
    await mongo_client.drop_database(settings.database)


def test_create_returns_24_hex_string_id(client):
    response = client.post("/items", json={"name": "widget"})
    assert response.status_code == 200
    item_id = response.json()["_id"]
    assert isinstance(item_id, str)
    assert ObjectId.is_valid(item_id)


def test_round_trip_id_survives_database_and_response(client):
    response = client.post("/items", json={"name": "widget"})
    item_id = response.json()["_id"]

    fetched = client.get(f"/items/{item_id}")
    assert fetched.status_code == 200
    assert fetched.json()["_id"] == item_id
    assert fetched.json()["name"] == "widget"


def test_malformed_id_returns_404_not_500(client):
    assert client.get("/items/not-an-object-id").status_code == 404
    assert client.get("/items/507f1f77bcf86cd79943901z").status_code == 404


def test_unknown_id_returns_404(client):
    unknown = str(ObjectId())
    assert client.get(f"/items/{unknown}").status_code == 404


async def test_stored_id_is_objectid_not_string(client, db):
    """Python-mode dumps insert a real ObjectId, so reads by id do not 404."""
    item_id = client.post("/items", json={"name": "widget"}).json()["_id"]
    doc = await db.items.find_one({"_id": ObjectId(item_id)})
    assert doc is not None
    assert isinstance(doc["_id"], ObjectId)
    assert doc["_id"] == ObjectId(item_id)


def test_get_client_without_lifespan_raises():
    from fastapi import Request

    request = Request({"type": "http", "app": FastAPI()})
    with pytest.raises(RuntimeError, match="lifespan=mongo_lifespan"):
        get_client(request)


def test_client_carries_handshake_driver_metadata(app):
    """The lifespan-created client must carry attribution metadata for Atlas.

    Asserts the DriverInfo passed at construction. Whether it reaches the
    server is pymongo's handshake contract; verified manually via
    $currentOp.clientMetadata.driver.name ("PyMongo|c|async|fastapi-pymongo")
    against a live deployment, but idle-connection listing in $currentOp is
    not consistent across environments, so it is not asserted here.
    """
    from importlib.metadata import version

    from fastapi_pymongo.lifespan import _STATE_KEY

    with TestClient(app):
        client = getattr(app.state, _STATE_KEY)
        driver = client.options._options["driver"]
        assert driver.name == "fastapi-pymongo"
        assert driver.version == version("fastapi-pymongo")


# --- cursor pagination tests -------------------------------------------------


async def _seed(db, documents):
    """Replace the collection's contents with ``documents``."""
    await db.items.delete_many({})
    if documents:
        await db.items.insert_many(documents)


async def _page_all(db, *, size, **kwargs):
    """Walk every page with :func:`cursor_page` and return the documents seen."""
    seen = []
    params = PageParams(size=size)
    while True:
        page = await cursor_page(db.items, params, **kwargs)
        assert len(page.items) <= size
        if page.next_cursor is not None:
            assert len(page.items) == size
        seen.extend(page.items)
        if page.next_cursor is None:
            return seen
        params = PageParams(cursor=page.next_cursor, size=size)


async def test_cursor_page_walks_all_documents_once_in_order(db):
    # An exact match against range(n) also proves no duplicates and no gaps.
    await _seed(db, [{"n": i} for i in range(10)])
    seen = [d["n"] for d in await _page_all(db, size=3, sort_field="n")]
    assert seen == list(range(10))


async def test_cursor_page_breaks_ties_on_id(db):
    # Every document shares one sort value, so the _id tiebreaker is the only
    # thing keeping the walk stable and complete.
    await _seed(db, [{"group": "x"} for _ in range(5)])
    docs = await _page_all(db, size=2, sort_field="group")
    ids = [d["_id"] for d in docs]
    assert len(ids) == 5
    assert len(set(ids)) == 5
    assert ids == sorted(ids)


async def test_cursor_page_descending(db):
    await _seed(db, [{"n": i} for i in range(5)])
    docs = await _page_all(db, size=2, sort_field="n", direction=DESCENDING)
    assert [d["n"] for d in docs] == [4, 3, 2, 1, 0]


async def test_cursor_page_respects_filter(db):
    await _seed(db, [{"n": i, "keep": i % 2 == 0} for i in range(6)])
    docs = await _page_all(db, size=2, sort_field="n", filter={"keep": True})
    assert [d["n"] for d in docs] == [0, 2, 4]


async def test_cursor_page_supports_bson_sort_values(db):
    base = datetime(2026, 1, 1)
    await _seed(db, [{"at": base + timedelta(days=i)} for i in range(4)])
    docs = await _page_all(db, size=2, sort_field="at")
    assert [d["at"] for d in docs] == [base + timedelta(days=i) for i in range(4)]


async def test_cursor_page_last_page_has_no_cursor(db):
    await _seed(db, [{"n": i} for i in range(2)])
    page = await cursor_page(db.items, PageParams(size=5), sort_field="n")
    assert [d["n"] for d in page.items] == [0, 1]
    assert page.next_cursor is None


async def test_cursor_page_rejects_bad_token(db):
    with pytest.raises(InvalidCursor):
        await cursor_page(db.items, PageParams(cursor="garbage"), sort_field="n")


async def test_cursor_page_rejects_cursor_from_other_direction(db):
    # A cursor records the direction it was issued with, so reusing it for a
    # walk in the opposite direction is rejected rather than silently wrong.
    await _seed(db, [{"n": i} for i in range(6)])
    page = await cursor_page(db.items, PageParams(size=2), sort_field="n")
    with pytest.raises(InvalidCursor):
        await cursor_page(
            db.items,
            PageParams(cursor=page.next_cursor, size=2),
            sort_field="n",
            direction=DESCENDING,
        )


async def test_cursor_page_exact_multiple_stops_cleanly(db):
    await _seed(db, [{"n": i} for i in range(6)])
    first = await cursor_page(db.items, PageParams(size=3), sort_field="n")
    second = await cursor_page(
        db.items, PageParams(cursor=first.next_cursor, size=3), sort_field="n"
    )
    assert [d["n"] for d in first.items] == [0, 1, 2]
    assert first.next_cursor is not None
    assert [d["n"] for d in second.items] == [3, 4, 5]
    assert second.next_cursor is None


async def test_cursor_page_empty_collection(db):
    await _seed(db, [])
    page = await cursor_page(db.items, PageParams(size=5), sort_field="n")
    assert page.items == []
    assert page.next_cursor is None


async def test_cursor_page_handles_null_sort_values(db):
    # Nulls sort first ascending; the walk must not stop at the null boundary.
    await _seed(db, [{"n": None}, {"n": 1}, {"n": None}, {"n": 2}])
    docs = await _page_all(db, size=1, sort_field="n")
    assert [d["n"] for d in docs] == [None, None, 1, 2]


async def test_cursor_page_handles_missing_sort_field(db):
    # Missing fields sort as null. Building the cursor used to raise KeyError.
    await _seed(db, [{"n": 2}, {"other": True}, {"n": 1}, {"other": True}])
    docs = await _page_all(db, size=1, sort_field="n")
    assert [d.get("n") for d in docs] == [None, None, 1, 2]


async def test_cursor_page_descending_with_null_sort_values(db):
    # Nulls sort last descending, so they are still reached after every value.
    await _seed(db, [{"n": None}, {"n": 1}, {"n": 3}])
    docs = await _page_all(db, size=1, sort_field="n", direction=DESCENDING)
    assert [d["n"] for d in docs] == [3, 1, None]


async def test_cursor_page_supports_dotted_sort_field(db):
    # A dotted path into an embedded document must resolve when the next
    # cursor is built, or the walk repeats the first page forever.
    await _seed(db, [{"meta": {"rank": i}} for i in range(5)])
    docs = await _page_all(db, size=2, sort_field="meta.rank")
    assert [d["meta"]["rank"] for d in docs] == [0, 1, 2, 3, 4]


async def test_cursor_page_descending_dotted_sort_field(db):
    await _seed(db, [{"meta": {"rank": i}} for i in range(4)])
    docs = await _page_all(db, size=2, sort_field="meta.rank", direction=DESCENDING)
    assert [d["meta"]["rank"] for d in docs] == [3, 2, 1, 0]


async def test_cursor_page_accepts_id_plus_exclusion_projection(db):
    # ``{"_id": 1, "secret": 0}`` is an exclusion projection (everything but
    # ``secret``); the sort field still comes back whole, so accept it.
    await _seed(db, [{"n": i, "secret": "hidden"} for i in range(4)])
    docs = await _page_all(db, size=2, sort_field="n", projection={"_id": 1, "secret": 0})
    assert [d["n"] for d in docs] == [0, 1, 2, 3]
    assert all("secret" not in d for d in docs)


async def test_cursor_page_rejects_operator_projection_on_sort_field(db):
    # ``$slice`` is truthy but returns a truncated value, so the next cursor
    # would be built from the wrong sort value; reject it up front.
    await _seed(db, [{"n": [i, i + 1]} for i in range(4)])
    with pytest.raises(ValueError, match="projection must return"):
        await cursor_page(
            db.items,
            PageParams(size=2),
            sort_field="n",
            projection={"n": {"$slice": 1}},
        )


async def test_cursor_page_rejects_sort_field_traversing_array(db):
    # Sorting by ``"tags.rank"`` over an array orders by the array's minimum
    # element, which the seek cannot reproduce, so the walk is rejected.
    await _seed(db, [{"tags": [{"rank": i}, {"rank": i + 10}]} for i in range(4)])
    with pytest.raises(ValueError, match="traverses an array"):
        await cursor_page(db.items, PageParams(size=2), sort_field="tags.rank")


async def test_cursor_page_rejects_array_sort_field_off_the_page_boundary(db):
    # The array document sorts first, so each page ends on a scalar. Validation
    # must cover every document, not just the boundary.
    await _seed(
        db,
        [
            {"tags": [{"rank": 1}, {"rank": 10}]},
            {"tags": {"rank": 2}},
            {"tags": {"rank": 3}},
            {"tags": {"rank": 4}},
        ],
    )
    with pytest.raises(ValueError, match="traverses an array"):
        await cursor_page(db.items, PageParams(size=2), sort_field="tags.rank")


async def test_cursor_page_accepts_slice_projection_on_unrelated_field(db):
    # A standalone $slice is an exclusion projection: unrelated fields,
    # including the sort field, still come back whole, so it must be accepted.
    await _seed(db, [{"n": i, "comments": [i, i + 1, i + 2]} for i in range(4)])
    docs = await _page_all(
        db, size=2, sort_field="n", projection={"comments": {"$slice": 1}}
    )
    assert [d["n"] for d in docs] == [0, 1, 2, 3]


async def test_cursor_page_handles_null_id(db):
    # _id may be null (at most one document). With equal sort values, an
    # ascending walk must not drop the rows that follow the null _id.
    await _seed(
        db, [{"_id": None, "n": 1}, {"_id": ObjectId(), "n": 1}, {"_id": ObjectId(), "n": 1}]
    )
    docs = await _page_all(db, size=1, sort_field="n")
    assert len(docs) == 3
    assert len({d["_id"] for d in docs}) == 3


async def test_cursor_page_default_id_sort_handles_null_id(db):
    # The same null-_id case on the default _id sort, where _id is the sort
    # field as well as the tiebreaker.
    await _seed(db, [{"_id": None, "n": 0}, {"_id": ObjectId(), "n": 1}])
    docs = await _page_all(db, size=1)
    assert len(docs) == 2


async def test_cursor_page_descending_handles_null_id_tiebreaker(db):
    # Descending, the null _id sorts last in its group; range comparisons are
    # type-bracketed, so the seek must add it explicitly.
    await _seed(
        db, [{"_id": None, "n": 1}, {"_id": ObjectId(), "n": 1}, {"_id": ObjectId(), "n": 1}]
    )
    docs = await _page_all(db, size=1, sort_field="n", direction=DESCENDING)
    assert len(docs) == 3
    assert len({d["_id"] for d in docs}) == 3


async def test_cursor_page_default_id_sort_descending_handles_null_id(db):
    # Same descending null-_id case on the default _id sort.
    await _seed(db, [{"_id": None, "n": 0}, {"_id": ObjectId(), "n": 1}])
    docs = await _page_all(db, size=1, direction=DESCENDING)
    assert len(docs) == 2


async def test_cursor_page_handles_mixed_type_ids(db):
    # _id values of different BSON types still sort by BSON order, so the
    # tiebreaker must compare across types; a type-bracketed $gt would stop
    # early after the integer id and drop the string one.
    await _seed(db, [{"_id": 1, "n": 5}, {"_id": "x", "n": 5}])
    docs = await _page_all(db, size=1, sort_field="n")
    assert {d["_id"] for d in docs} == {1, "x"}


async def test_cursor_page_descending_handles_mixed_type_ids(db):
    await _seed(db, [{"_id": 1, "n": 5}, {"_id": "x", "n": 5}])
    docs = await _page_all(db, size=1, sort_field="n", direction=DESCENDING)
    assert {d["_id"] for d in docs} == {1, "x"}


async def test_cursor_page_rejects_computed_literal_projection(db):
    # An array literal is a computed value that replaces the sort field, not an
    # inclusion flag, so it must be rejected.
    await _seed(db, [{"n": i} for i in range(4)])
    with pytest.raises(ValueError, match="projection must return"):
        await cursor_page(
            db.items, PageParams(size=2), sort_field="n", projection={"n": [1, 2]}
        )


async def test_cursor_page_rejects_numeric_string_projection(db):
    # A numeric string is an expression that replaces the stored value, so it
    # must not count as returning the sort field.
    await _seed(db, [{"n": i} for i in range(4)])
    with pytest.raises(ValueError, match="projection must return"):
        await cursor_page(
            db.items, PageParams(size=2), sort_field="n", projection={"n": "1"}
        )


async def test_cursor_page_accepts_sort_field_nested_under_included_id(db):
    # {"_id": 1} returns a document-valued _id whole, so a sort field nested
    # under it is present and the projection is valid.
    await _seed(db, [{"_id": {"rank": i}, "n": i} for i in range(4)])
    docs = await _page_all(db, size=2, sort_field="_id.rank", projection={"_id": 1})
    assert [d["_id"]["rank"] for d in docs] == [0, 1, 2, 3]


async def test_cursor_page_does_not_run_forged_sort_value_as_operator(db):
    # A forged cursor value shaped like a query operator must be compared
    # literally, not executed as a predicate.
    await _seed(db, [{"n": "aaa"}, {"n": "bbb"}])
    forged = _encode_cursor(
        {"$regex": "a"}, ObjectId("000000000000000000000000"), ASCENDING
    )
    page = await cursor_page(db.items, PageParams(cursor=forged, size=5), sort_field="n")
    assert page.items == []


async def test_cursor_page_supports_uuid_sort_values(db):
    # A collection with uuidRepresentation=standard returns uuid.UUID values;
    # the token must encode them with the collection's codec options or the
    # walk fails as soon as a next-page token is built.
    options = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
    collection = db.get_collection("uuid_items", codec_options=options)
    await collection.delete_many({})
    await collection.insert_many([{"n": i, "u": uuid4()} for i in range(4)])
    seen = []
    params = PageParams(size=2)
    while True:
        page = await cursor_page(collection, params, sort_field="u")
        seen.extend(d["u"] for d in page.items)
        if page.next_cursor is None:
            break
        params = PageParams(cursor=page.next_cursor, size=2)
    assert len(seen) == 4
    assert len(set(seen)) == 4


@pytest.mark.xfail(
    strict=True,
    reason="keyset seek compares within one BSON type; see the pagination docs",
)
async def test_cursor_page_mixed_sort_types_stop_early(db):
    # Documents a known limitation: non-null sort values of different BSON
    # types do not page correctly. If this starts passing, remove the xfail and
    # the "one BSON type" caveat from the docs.
    await _seed(db, [{"n": 1}, {"n": "a"}, {"n": 2}])
    docs = await _page_all(db, size=1, sort_field="n")
    assert len(docs) == 3


@pytest.mark.xfail(
    strict=True,
    reason="arrays sort by min element but the seek compares lexicographically; see the pagination docs",
)
async def test_cursor_page_array_sort_values_duplicate(db):
    # Documents a known limitation: MongoDB orders an array-valued sort field
    # by its minimum element, while the seek compares arrays lexicographically,
    # so the walk can repeat a document. If this starts passing, remove the
    # xfail and the "array-valued sort fields" caveat from the docs.
    await _seed(db, [{"tags": [5, 1]}, {"tags": [2]}, {"tags": [3, 9]}])
    docs = await _page_all(db, size=2, sort_field="tags")
    assert len(docs) == 3


async def test_cursor_page_sorts_by_subdocument_when_returned_whole(db):
    # A subdocument sort field is compared field by field, so a projection that
    # returns the whole subdoc pages correctly.
    await _seed(db, [{"meta": {"rank": i, "other": i}} for i in range(5)])
    docs = await _page_all(db, size=2, sort_field="meta", projection={"meta": 1})
    assert [d["meta"]["rank"] for d in docs] == [0, 1, 2, 3, 4]


async def test_cursor_page_rejects_partial_subdocument_projection(db):
    # Projecting only a child of the sort field returns a truncated subdoc,
    # whose cursor would re-match rows (duplicates). It is rejected up front,
    # before the query, so the collection is not touched.
    await _seed(db, [{"meta": {"rank": i, "other": i}} for i in range(5)])
    with pytest.raises(ValueError, match="projection must return"):
        await cursor_page(
            db.items,
            PageParams(size=2),
            sort_field="meta",
            projection={"meta.rank": 1},
        )
