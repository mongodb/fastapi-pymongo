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

Offset pagination (``skip``/``limit``) is easy, but it has two problems:
``skip`` is an O(n) scan on the server, so deep pages get slower, and
concurrent inserts or deletes shift the window, so clients see duplicates or
miss documents. Cursor pagination avoids both by seeking past
the last document's sort key rather than counting from the top.

This module provides cursor pagination for a FastAPI + ``AsyncMongoClient``
app.

How it works
------------

Documents are ordered by ``sort_field``, breaking ties on ``_id``.
Null and missing sort values are ordered as MongoDB orders them — first
ascending, last descending — and the seek handles that boundary, so a sort
field that is only sometimes present still pages without dropping documents.
The direction (ascending or descending) is validated on decode: reusing a cursor with the opposite
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

``cursor_page`` returns raw PyMongo documents. Declaring a response model
serializes ``_id`` through ``PyObjectId`` to prevent an encoding failure on ``ObjectId``.

``size`` defaults to 20 and has no upper bound; a page is fetched in full, so
bound ``size`` in your app if the endpoint is public.

Indexing
--------

For a bounded index scan rather than a collection scan, index the sort keys
in the order used:

.. code-block:: python

   from pymongo import DESCENDING

   await db.items.create_index(
       [("created_at", DESCENDING), ("_id", DESCENDING)]
   )

That index fits a descending ``created_at`` pagination. The pagination examples above use
the default, ascending, direction, so index ``ASCENDING`` there. Match the
leading direction to the sort you use most.

What this does not do
---------------------

- **No compound sorts.** For a compound sort, write the query yourself.
- **Not encrypted.** The ``next_cursor`` token is base64url-encoded BSON, not
  ciphertext, so anyone who sees it can decode the ``_id`` and sort value with
  no key.
- **Not signed.** The ``next_cursor`` token is not tamper-proof: a client can
  forge one.
