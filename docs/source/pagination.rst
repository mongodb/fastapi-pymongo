Cursor pagination
=================

.. currentmodule:: fastapi_pymongo.pagination

.. autofunction:: cursor_page
.. autofunction:: page_params
.. autoclass:: PageParams
.. autoclass:: CursorPage
.. autoexception:: InvalidCursor

Why
---

Offset pagination (``skip``/``limit``) is easy but degrades. ``skip`` is an
O(n) scan on the server, and concurrent inserts or deletes shift the window,
so clients see duplicates or miss documents. Cursor (keyset) pagination
avoids both by seeking from the last document you saw instead of counting
from the top.

This module provides keyset pagination for a FastAPI + ``AsyncMongoClient``
app, without an ODM.

How it works
------------

Documents are ordered by ``sort_field``, breaking ties on ``_id`` (so just
``_id`` when ``sort_field`` is ``_id``). The ``_id`` tiebreaker gives a stable
total order even when the sort field repeats, so a walk neither skips nor
duplicates a document — provided the sort values meet the conditions under
`What this does not do`_. The next-page token is the last document's
``(sort_value, _id)`` pair plus the sort direction, BSON-encoded and
base64url-encoded, so any value MongoDB can store round-trips. It carries no
query state and clients should treat it as opaque — but it is not encrypted
(see `What this does not do`_).

Null and missing sort values are ordered as MongoDB orders them — first
ascending, last descending — and the seek handles that boundary, so a sort
field that is only sometimes present still pages without dropping documents.
The direction is validated on decode: reusing a cursor with the opposite
direction raises :class:`InvalidCursor` rather than returning the wrong slice.

Usage
-----

A runnable version is in
`examples/pagination/main.py <https://github.com/mongodb/fastapi-pymongo/blob/main/examples/pagination/main.py>`_.

.. code-block:: python

   from fastapi import Depends, FastAPI
   from pydantic import BaseModel, ConfigDict, Field
   from pymongo.asynchronous.database import AsyncDatabase

   from fastapi_pymongo import (
       MongoSettings,
       PageParams,
       PyObjectId,
       cursor_page,
       mongo_lifespan,
       page_params,
   )
   from fastapi_pymongo.lifespan import get_database

   settings = MongoSettings()
   app = FastAPI(lifespan=mongo_lifespan(settings))
   db_dependency = get_database(settings)


   class Item(BaseModel):
       model_config = ConfigDict(populate_by_name=True)

       id: PyObjectId = Field(alias="_id", default_factory=PyObjectId)
       name: str


   class ItemPage(BaseModel):
       items: list[Item]
       next_cursor: str | None = None


   @app.get("/items", response_model=ItemPage)
   async def list_items(
       params: PageParams = Depends(page_params),
       db: AsyncDatabase = Depends(db_dependency),
   ):
       page = await cursor_page(db.items, params, sort_field="name")
       return {"items": page.items, "next_cursor": page.next_cursor}

``cursor_page`` returns raw PyMongo documents. Declare a response model (as
above) so ``_id`` serializes through ``PyObjectId``; otherwise FastAPI cannot
encode ``ObjectId`` and the route fails.

``page_params`` declares ``?cursor=`` and ``?size=`` as query parameters, so
they appear in the generated OpenAPI docs. The client sends the
``next_cursor`` from one response as ``?cursor=`` on the next request.
``next_cursor`` is ``null`` on the final page.

``size`` defaults to 20 and has no upper bound; a page is fetched in full, so
bound ``size`` in your app if the endpoint is public.

Invalid cursors
---------------

A token that was not produced by this library raises :class:`InvalidCursor`,
a :class:`fastapi.HTTPException` with status ``400``. A route that lets it
propagate returns that 400 with a generic ``detail`` (the token is not echoed
back), so no handler is needed; register one only to customize the response.

Indexing
--------

For a bounded index scan rather than a collection scan, index the sort keys
in the order used:

.. code-block:: python

   from pymongo import DESCENDING

   await db.items.create_index(
       [("created_at", DESCENDING), ("_id", DESCENDING)]
   )

That index fits a descending ``created_at`` walk. The walk examples above use
the default, ascending, direction, so index ``ASCENDING`` there. Either way is
cheap — MongoDB scans an index in reverse just as well — so match the leading
direction to the sort you use most and reuse the index for both.

What this does not do
---------------------

- **No query builder.** ``filter`` is passed to ``find`` unchanged; this is
  not a place to compose queries or joins.
- **One sort field.** Order by a single field plus the ``_id`` tiebreaker.
  For a compound sort, write the query yourself. A dotted path into an
  embedded document (``"meta.rank"``) is a single field and works; a path that
  traverses an array is rejected (see below).
- **Non-null sort values must share one BSON type.** The seek uses
  ``$gt``/``$lt``, which only compare within a single BSON type (numbers
  compare together; other types do not cross). A field holding, say, both
  strings and numbers will stop early. Null and missing are handled; genuinely
  mixed non-null types are not. ``_id`` is exempt: the tiebreaker compares by
  BSON order, so ids of different types still page correctly.
- **Array-valued sort fields are not supported.** MongoDB orders an array
  field by its minimum element, but the seek compares arrays
  lexicographically, so a walk over array values can repeat documents. Sort
  by a scalar field instead. For the same reason a dotted path that traverses
  an array by name (``"tags.rank"`` where ``tags`` is a list of subdocuments)
  is rejected with :exc:`ValueError` rather than paged incorrectly.
- **The sort field is not in the token.** Only the direction is, and it is
  validated on decode. Keep ``sort_field`` constant across a walk, or the
  cursor is meaningless.
- **Not encrypted.** The token is base64url-encoded BSON, not ciphertext, so
  anyone who sees it can decode the ``_id`` and sort value with no key. Those
  values are already in the response body, but the token also travels in URLs,
  access logs, and ``Referer`` headers, so avoid sorting on a field whose value
  is sensitive.
- **Not signed.** The token is opaque but not tamper-proof: a client can
  forge one. It only affects which slice of data comes back, so that is
  usually acceptable; sign or encrypt it yourself if it is not. A forged
  value is still compared literally — a sort value shaped like a query
  operator (``{"$regex": ...}``) is matched as a value, never executed.
- **Projections must return ``sort_field`` and ``_id`` in full.**
  ``cursor_page`` raises :exc:`ValueError` otherwise, since a missing or partial
  value builds a silently wrong cursor. That includes a child path of a
  subdocument field, an operator that truncates it (``$slice``, ``$elemMatch``),
  an expression that replaces it (``{"n": "$other"}``), and a descendant or
  computed ``_id``. An exclusion projection is fine: ``{"_id": 1, "secret": 0}``
  and a standalone ``$slice`` on an unrelated field are both accepted.
