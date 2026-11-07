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

    Covers a token this library did not produce and one issued for a different
    sort direction. It is a :class:`fastapi.HTTPException` with status ``400``,
    so a route that lets it propagate needs no handler (see :doc:`/pagination`).
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
    """Encode ``(sort_value, _id, direction)`` as an opaque, URL-safe token.

    The payload is BSON, so any storable value round-trips. ``codec_options``
    are the collection's, so codec-dependent values (e.g. ``uuid.UUID``) encode
    as they were read.
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
    """Decode a token into ``(sort_value, doc_id, direction)``.

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
    """Return the value at dotted ``path`` in ``document``, or ``None``.

    A missing key, a non-mapping intermediate, or an out-of-range index resolves
    to ``None`` (MongoDB treats a missing field as null).

    :raises ValueError: if ``path`` crosses an array by name, whose min-element
        ordering the keyset seek cannot reproduce.
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
            # A named segment crossing an array: MongoDB sorts by the array's
            # minimum, not this value.
            raise ValueError(
                f"sort field {path!r} traverses an array; array-valued sort "
                "fields are not supported"
            )
        else:
            return None
    return value


def _is_whole_field_flag(value: Any) -> bool:
    """Return whether a projection entry includes its field in full.

    Only a truthy scalar (``1``, ``true``, a non-zero number) is an inclusion
    flag. A string is an expression that replaces the field (``{"n": "1"}``
    sets ``n`` to ``"1"``), and an operator or falsy value does not include it.
    """
    if isinstance(value, (Mapping, str)):
        return False
    return bool(value)


def _is_inclusion_entry(value: Any) -> bool:
    """Return whether a projection entry selects inclusion mode.

    Include flags, expressions (any string), and operators other than ``$slice``
    are inclusion; exclude flags and a standalone ``$slice`` are exclusion.
    """
    if isinstance(value, Mapping):
        return "$slice" not in value
    if isinstance(value, str):
        return True
    return bool(value)


def _projection_returns_fully(projection: Mapping[str, Any] | None, field: str) -> bool:
    """Return whether ``projection`` returns ``field`` in full.

    A field is whole only if its whole value comes back: an inclusion projection
    must include it (or an ancestor) with a flag (``{"meta": 1}`` keeps
    ``meta.rank`` whole; ``{"meta.rank": 1}`` does not keep ``meta``), and an
    exclusion projection must not exclude or ``$slice`` anything overlapping it.
    ``_id`` is included by default, so it and any field nested under it are whole
    unless the ``_id`` subtree is excluded, computed, or truncated.

    The form is decided by the fields other than ``_id`` (which may appear in
    either): ``{"_id": 1, "secret": 0}`` is an exclusion projection.
    """
    if projection is None:
        return True
    others = {path: value for path, value in projection.items() if path != _ID}
    inclusion = (
        any(_is_inclusion_entry(value) for value in others.values())
        if others
        else _is_inclusion_entry(projection.get(_ID, 0))
    )
    # ``_id`` is included whole by default, so a field nested under it is
    # present unless the projection touches the ``_id`` subtree; model that with
    # an implicit ``{"_id": 1}``.
    paths = projection
    if not any(path == _ID or path.startswith(f"{_ID}.") for path in projection):
        paths = {**projection, _ID: 1}
    if inclusion:
        return any(
            (field == path or field.startswith(f"{path}."))
            and _is_whole_field_flag(value)
            for path, value in paths.items()
        )
    return not any(
        field == path
        or field.startswith(f"{path}.")
        or path.startswith(f"{field}.")
        for path, value in paths.items()
        if not _is_whole_field_flag(value)
    )


def _id_seek(direction: int, last_id: Any) -> dict[str, Any]:
    """Build the ``_id`` tiebreaker seek past ``last_id``.

    ``_id`` is unique, so at most one document has a null ``_id``. Range
    comparisons are type-bracketed and never match it: ascending it sorts first
    (all non-null ids follow), descending it sorts last (so add it explicitly).
    """
    if last_id is None:
        if direction == ASCENDING:
            return {_ID: {"$ne": None}}
        # The null id is last descending, so nothing follows it.
        return {_ID: {"$lt": None}}
    if direction == ASCENDING:
        return {_ID: {"$gt": last_id}}
    return {"$or": [{_ID: {"$lt": last_id}}, {_ID: None}]}


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
        # Nulls sort lowest, and ``{field: {"$gt": None}}`` matches nothing, so
        # handle them explicitly (range comparisons are type-bracketed).
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
        # Seek past the last sort value, tie-broken on _id (a total order). $eq
        # keeps a cursor value like ``{"$regex": ...}`` literal, not a predicate.
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
    ``_id`` when ``sort_field`` is ``_id``), giving a stable total order. To read
    the next page, pass ``next_cursor`` back as ``PageParams.cursor``; ``None``
    means the end.

    The token records its sort direction, so a cursor from one direction is
    rejected if reused with the other.

    Null and missing sort values page correctly, as does a null ``_id``.
    Non-null sort values must share one BSON type: the seek's ``$gt``/``$lt`` do
    not compare across types.

    :param collection: the PyMongo async collection to read from.
    :param params: the page cursor and size, usually from :func:`page_params`.
    :param sort_field: the field to order by (defaults to ``_id``). A dotted
        path into an embedded document is supported; a path that traverses an
        array is rejected with :exc:`ValueError`.
    :param direction: ``pymongo.ASCENDING`` (default) or
        ``pymongo.DESCENDING``.
    :param filter: an optional MongoDB query, passed through unchanged.
    :param projection: an optional projection. It must return ``sort_field`` and
        ``_id`` in full (a subdocument ``sort_field`` must come back whole), or
        :exc:`ValueError` is raised.
    :raises InvalidCursor: if ``params.cursor`` is malformed, or was issued
        for a different sort direction.
    :raises ValueError: if ``direction`` is invalid, ``params.size`` is below 1,
        ``projection`` does not return ``sort_field`` or ``_id`` in full, or
        ``sort_field`` traverses an array.
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
        last_value, last_id, cursor_direction = _decode_cursor(
            params.cursor, codec_options
        )
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
    # Resolve every returned document, not just the cursor's: an array traversal
    # anywhere makes the ordering unreproducible, so it must raise wherever it
    # lands. A missing sort field resolves to null, which MongoDB orders as null.
    resolved = [_resolve_field(document, sort_field) for document in items]
    next_cursor = None
    if has_more and items:
        next_cursor = _encode_cursor(
            resolved[-1], items[-1][_ID], direction, codec_options
        )
    return CursorPage(items=items, next_cursor=next_cursor)
