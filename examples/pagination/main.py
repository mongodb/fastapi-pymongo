"""Cursor-paginated listing: FastAPI + PyMongo Async, via fastapi-pymongo.

Run:
    pip install -e ".[dev]"
    uvicorn examples.pagination.main:app --reload

Create a few items, then walk the pages with the ``next_cursor`` from each
response:

    curl -s -X POST localhost:8000/items -H 'content-type: application/json' \\
        -d '{"name": "widget", "price": 1.5}'
    curl -s 'localhost:8000/items?size=2'
    curl -s 'localhost:8000/items?size=2&cursor=<next_cursor>'
"""

from __future__ import annotations

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
    price: float


class ItemPage(BaseModel):
    items: list[Item]
    next_cursor: str | None = None


@app.post("/items", response_model=Item)
async def create_item(item: Item, db: AsyncDatabase = Depends(db_dependency)):
    doc = item.model_dump(by_alias=True)
    await db.items.insert_one(doc)
    return doc


@app.get("/items", response_model=ItemPage)
async def list_items(
    params: PageParams = Depends(page_params),
    db: AsyncDatabase = Depends(db_dependency),
):
    # cursor_page returns raw PyMongo documents; the response model turns each
    # _id into a PyObjectId so FastAPI can serialize it. next_cursor is null
    # on the last page, and an invalid cursor returns 400 on its own.
    page = await cursor_page(db.items, params, sort_field="name")
    return {"items": page.items, "next_cursor": page.next_cursor}
