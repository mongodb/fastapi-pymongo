"""Cursor (keyset) pagination for raw PyMongo Async collections.

:func:`cursor_page` returns one page of documents plus an opaque token for
the next page. It is deliberately not a query builder or an ODM: you pass a
collection and a filter, and it returns documents.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from bson import BSON
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions
from bson.errors import InvalidBSON
from fastapi import HTTPException, Query
from pymongo import ASCENDING, DESCENDING
from pymongo.asynchronous.collection import AsyncCollection

DEFAULT_PAGE_SIZE = 20

_ID = "_id"
_BASE64_GROUP_SIZE = 4


class InvalidCursor(HTTPException):
    """Raised when a cursor token cannot be used for the request.

    That covers a token this library did not produce and one it did produce
    but issued for a different sort direction. It is an
    :class:`fastapi.HTTPException` with status ``400``, so a route that lets
    it propagate returns a 400 without any handler. Register one only to
    customize the response; see :doc:`/pagination`.
    """

    def __init__(self, detail: str = "Invalid cursor") -> None:
        super().__init__(status_code=400, detail=detail)


@dataclass(frozen=True)
class PageParams:
    """Pagination inputs parsed from the query string.

    :param cursor: opaque token from the previous page's ``next_cursor``, or
        ``None`` for the first page.
    :param size: maximum number of documents per page.
    """

    cursor: str | None = None
    size: int = DEFAULT_PAGE_SIZE


@dataclass(frozen=True)
class CursorPage:
    """One page of results.

    :param items: the documents in this page.
    :param next_cursor: token to pass back as ``PageParams.cursor`` for the
        next page, or ``None`` when this is the last page.
    """

    items: list[Mapping[str, Any]]
    next_cursor: str | None = None


def _encode_cursor(
    sort_value: Any,
    doc_id: Any,
    direction: int,
    codec_options: CodecOptions[Any] = DEFAULT_CODEC_OPTIONS,
) -> str:
    """Encode a ``(sort_value, _id, direction)`` triple as an opaque, URL-safe token.

    The payload is BSON-encoded, so any value MongoDB can store round-trips
    (``ObjectId``, ``datetime``, ``Decimal128``, embedded documents, ...).
    ``codec_options`` are the source collection's, so values whose encoding
    depends on them (notably ``uuid.UUID`` under ``uuidRepresentation``)
    encode the same way they were read.
    """
    payload = BSON.encode(
        {"sort_value": sort_value, "_id": doc_id, "direction": direction},
        codec_options=codec_options,
    )
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _decode_cursor(
    token: str,
    codec_options: CodecOptions[Any] = DEFAULT_CODEC_OPTIONS,
) -> tuple[Any, Any, int]:
    """Decode a token produced by :func:`_encode_cursor`.

    Returns ``(sort_value, doc_id, direction)``.

    :raises InvalidCursor: if the token is malformed.
    """
    padded = token + "=" * (-len(token) % _BASE64_GROUP_SIZE)
    try:
        document = BSON(base64.urlsafe_b64decode(padded)).decode(
            codec_options=codec_options
        )
        return document["sort_value"], document["_id"], document["direction"]
    except (InvalidBSON, KeyError, ValueError) as exc:
        # binascii.Error is a ValueError, so bad base64 is covered here too.
        raise InvalidCursor() from exc


def page_params(
    cursor: Annotated[
        str | None,
        Query(description="Opaque cursor from the previous page's next_cursor."),
    ] = None,
    size: Annotated[
        int,
        Query(
            ge=1,
            description="Maximum number of documents to return.",
        ),
    ] = DEFAULT_PAGE_SIZE,
) -> PageParams:
    """FastAPI dependency that parses ``?cursor=`` and ``?size=``.

    Use it with ``Depends`` so both parameters appear in the OpenAPI docs::

        async def list_items(params: PageParams = Depends(page_params)):
            ...
    """
    return PageParams(cursor=cursor, size=size)


def _resolve_field(document: Mapping[str, Any], path: str) -> Any:
    """Return the value at a dotted ``path`` within ``document``.

    MongoDB treats a missing field as null, so an unresolvable path (a missing
    key, a non-mapping at an intermediate step, or an out-of-range array index)
    returns ``None``. A plain field name is the single-segment case.

    :raises ValueError: if ``path`` traverses an array by name. MongoDB orders
        such a field by the minimum element of the array, which the keyset seek
        cannot reproduce, so it is rejected rather than paged incorrectly.
    """
    value: Any = document
    for part in path.split("."):
        if isinstance(value, Mapping):
            if part not in value:
                return None
            value = value[part]
        elif isinstance(value, list) and part.isdigit():
            index = int(part)
            if index >= len(value):
                return None
            value = value[index]
        elif isinstance(value, list):
            # A named segment crossing an array: MongoDB resolves it against
            # every element and sorts by the minimum, so the value here is not
            # the one the server ordered by.
            raise ValueError(
                f"sort field {path!r} traverses an array; array-valued sort "
                "fields are not supported"
            )
        else:
            return None
    return value


def _is_whole_field_flag(value: Any) -> bool:
    """Return whether a projection entry includes its field in full.

    A truthy scalar (``1``, ``true``, or a numeric string) is an inclusion
    flag. Anything else — a falsy scalar, an operator such as ``$slice`` or
    ``$elemMatch``, or an aggregation-style expression like ``"$other"`` —
    does not return the field whole.
    """
    if isinstance(value, Mapping):
        return False
    if isinstance(value, str):
        # MongoDB parses numeric strings as flags; other strings are
        # expressions that replace the field rather than returning it.
        try:
            return float(value) != 0
        except ValueError:
            return False
    return bool(value)


def _is_inclusion_entry(value: Any) -> bool:
    """Return whether a projection entry puts the projection in inclusion mode.

    Inclusion flags and operators other than ``$slice`` (``$elemMatch``,
    ``$meta``, expressions) are inclusion; exclude flags and a standalone
    ``$slice`` are exclusion.
    """
    if isinstance(value, Mapping):
        return "$slice" not in value
    if isinstance(value, str):
        try:
            return float(value) != 0
        except ValueError:
            return True
    return bool(value)


def _projection_returns_fully(projection: Mapping[str, Any] | None, field: str) -> bool:
    """Return whether ``projection`` returns ``field`` in full.

    This is stricter than a presence check: a subdocument sort field must come
    back whole for its cursor value to be correct, so projecting (or excluding)
    only a child path is not enough.

    Handles both MongoDB projection forms. An inclusion projection returns only
    the listed paths, plus ``_id`` unless it is explicitly excluded; a field is
    returned whole when it, or an ancestor, is included with a flag
    (``{"meta": 1}`` keeps ``meta.rank`` whole; ``{"meta.rank": 1}`` does not
    keep ``meta`` whole, and an operator or expression never keeps a field
    whole). An exclusion projection returns everything except the falsy paths
    and any ``$slice``, so a field is whole only when nothing overlapping it is
    excluded or sliced. ``_id`` is whole unless it is excluded, computed, or
    has a projected descendant.

    The projection form is decided by the fields other than ``_id``, because
    ``_id`` may appear in either form: ``{"_id": 1, "secret": 0}`` is an
    exclusion projection (everything but ``secret``), not an inclusion one.
    Only when ``_id`` is the sole entry does its own value pick the form.
    """
    if projection is None:
        return True
    if field == _ID:
        # A descendant like ``_id.part`` truncates a document-valued _id, and a
        # computed value (``{"$toString": "$_id"}``) is not the stored id at
        # all, so both break the tiebreaker.
        if any(path.startswith(f"{_ID}.") for path in projection):
            return False
        return _is_whole_field_flag(projection.get(_ID, 1))
    others = {path: value for path, value in projection.items() if path != _ID}
    inclusion = (
        any(_is_inclusion_entry(value) for value in others.values())
        if others
        else _is_inclusion_entry(projection.get(_ID, 0))
    )
    if inclusion:
        return any(
            (field == path or field.startswith(f"{path}."))
            and _is_whole_field_flag(value)
            for path, value in others.items()
        )
    return not any(
        field == path
        or field.startswith(f"{path}.")
        or path.startswith(f"{field}.")
        for path, value in others.items()
        if not _is_whole_field_flag(value)
    )


def _id_seek(direction: int, last_id: Any) -> dict[str, Any]:
    """Build the ``_id`` tiebreaker seek past ``last_id``.

    ``_id`` is unique, so at most one document has a null ``_id``: ascending it
    sorts first, so every non-null id follows; descending it sorts last, so
    nothing follows and the range comparison (``$lt: null``) matches nothing.
    """
    if last_id is None and direction == ASCENDING:
        return {_ID: {"$ne": None}}
    operator = "$gt" if direction == ASCENDING else "$lt"
    return {_ID: {operator: last_id}}


def _keyset_query(
    base: Mapping[str, Any],
    sort_field: str,
    direction: int,
    last_value: Any,
    last_id: Any,
) -> dict[str, Any]:
    """Build the "seek past the last document" condition for the next page."""
    operator = "$gt" if direction == ASCENDING else "$lt"
    if sort_field == _ID:
        seek: dict[str, Any] = _id_seek(direction, last_id)
    elif last_value is None:
        # Null and missing sort lowest in BSON order, and are handled
        # explicitly because ``{field: {"$gt": None}}`` matches nothing:
        # $gt/$lt only compare within a single BSON type.
        if direction == ASCENDING:
            # Every non-null value sorts after all the nulls.
            seek = {
                "$or": [
                    {sort_field: {"$ne": None}},
                    {"$and": [{sort_field: None}, _id_seek(direction, last_id)]},
                ]
            }
        else:
            # Nulls sort last, so only later nulls remain.
            seek = {"$and": [{sort_field: None}, _id_seek(direction, last_id)]}
    else:
        # Seek past the last sort value, and use _id to break ties when the
        # sort field repeats (keyset pagination needs a total order). The value
        # is wrapped in $eq so a cursor value like ``{"$regex": ...}`` is
        # compared literally instead of being run as a query operator.
        branches: list[dict[str, Any]] = [
            {sort_field: {operator: last_value}},
            {"$and": [{sort_field: {"$eq": last_value}}, _id_seek(direction, last_id)]},
        ]
        if direction == DESCENDING:
            # Nulls sort after every non-null value, so they still follow.
            branches.append({sort_field: None})
        seek = {"$or": branches}
    if not base:
        return seek
    return {"$and": [dict(base), seek]}


async def cursor_page(
    collection: AsyncCollection[Any],
    params: PageParams,
    *,
    sort_field: str = _ID,
    direction: int = ASCENDING,
    filter: Mapping[str, Any] | None = None,
    projection: Mapping[str, Any] | None = None,
) -> CursorPage:
    """Fetch one keyset-paginated page from ``collection``.

    Documents are ordered by ``sort_field``, breaking ties on ``_id`` (so just
    ``_id`` when ``sort_field`` is ``_id``). Equal sort values therefore still
    have a stable, total order. To read the next page, pass the returned
    ``next_cursor`` back as ``PageParams.cursor``; when ``next_cursor`` is
    ``None`` you have reached the end.

    The token records the sort direction it was issued with, so a cursor from
    a walk in one direction is rejected if reused with the other.

    Null and missing sort values are handled, so a sort field that is only
    sometimes present still pages correctly, and a null ``_id`` (which at most
    one document may have) is handled in the tiebreaker. Non-null sort values,
    however, must share one BSON type: the seek uses ``$gt``/``$lt``, which do
    not compare across BSON types.

    :param collection: the PyMongo async collection to read from.
    :param params: the page cursor and size, usually from :func:`page_params`.
    :param sort_field: the field to order by (defaults to ``_id``). A dotted
        path into an embedded document is supported; a path that traverses an
        array is rejected with :exc:`ValueError`.
    :param direction: ``pymongo.ASCENDING`` (default) or
        ``pymongo.DESCENDING``.
    :param filter: an optional MongoDB query, passed through unchanged.
    :param projection: an optional projection. It must return ``sort_field``
        and ``_id`` in full, or the next cursor cannot be built and
        :exc:`ValueError` is raised. A ``sort_field`` that is a subdocument
        must come back whole, so projecting only part of it is rejected.
    :raises InvalidCursor: if ``params.cursor`` is malformed, or was issued
        for a different sort direction.
    :raises ValueError: if ``direction`` is not one of the two PyMongo sort
        constants, ``params.size`` is less than 1, ``projection`` does not
        return ``sort_field`` or ``_id`` in full, or ``sort_field`` traverses
        an array.
    """
    if (
        not isinstance(direction, int)
        or isinstance(direction, bool)
        or direction not in (ASCENDING, DESCENDING)
    ):
        raise ValueError("direction must be pymongo.ASCENDING or pymongo.DESCENDING")
    if params.size < 1:
        raise ValueError("params.size must be at least 1")
    if projection is not None:
        dropped = [
            name
            for name in dict.fromkeys((sort_field, _ID))
            if not _projection_returns_fully(projection, name)
        ]
        if dropped:
            raise ValueError(
                f"projection must return {' and '.join(dropped)} in full "
                "to build the next cursor"
            )

    query: dict[str, Any] = dict(filter) if filter else {}
    codec_options = collection.codec_options
    if params.cursor is not None:
        last_value, last_id, cursor_direction = _decode_cursor(params.cursor, codec_options)
        if cursor_direction != direction:
            raise InvalidCursor("cursor was issued for a different sort direction")
        query = _keyset_query(query, sort_field, direction, last_value, last_id)

    if sort_field == _ID:
        sort_spec: list[tuple[str, int]] = [(_ID, direction)]
    else:
        sort_spec = [(sort_field, direction), (_ID, direction)]

    # Fetch one extra document to learn whether another page exists.
    cursor = collection.find(query, projection).sort(sort_spec).limit(params.size + 1)
    documents = await cursor.to_list(length=params.size + 1)

    has_more = len(documents) > params.size
    items = documents[: params.size]
    # Resolve the sort value for every returned document, not just the one used
    # to build the cursor: an array traversal anywhere in the page makes the
    # ordering unreproducible, so it must raise regardless of where it lands.
    # A document may be missing the sort field, which MongoDB orders as null.
    resolved = [_resolve_field(document, sort_field) for document in items]
    next_cursor = None
    if has_more and items:
        next_cursor = _encode_cursor(
            resolved[-1], items[-1][_ID], direction, codec_options
        )
    return CursorPage(items=items, next_cursor=next_cursor)
