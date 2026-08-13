---
title: "FastAPI"
icon: simple/fastapi
---

# FastAPI

![FastAPI-Logo](../assets/fastapi-logo.avif)

> FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.10+ based on standard Python type hints.

!!! info
    FastAPI builds on the idea that **type hints are the source of truth** for your code. It uses these type hints to validate, serialize, and document your API automatically.

## Setup

Two things are crucial when installing FastAPI: 

1. **Standard Versions:** the `fastapi[standard]` comes with all the recommended dependencies for production use.

2. **Pin Versions:** FastAPI is still in 0.x, so it is recommended to pin the version in your `pyproject.toml`. -> e.g. `fastapi>=139.0,<140.0`

## Basics

<div class="grid" markdown>
<div>
The decorators `@app.get()`, `@app.post()`, etc. are the *path operation functions*. Its return value is automatically converted to JSON. <br/>
You can run it with:

    ```bash
    fastapi dev main.py # dev-mode: auto-reload
    fastapi run main.py # prod-mode: no reload
    ```
</div>

!!! example

    ```python
    from fastapi import FastAPI

    app = FastAPI() # the ASGI application object

    # "when a GET request hits /, run this function"
    @app.get("/")
    def read_root():
        return {"message": "Hello World"}
    ```

</div>

Per default the FastAPI application runs on `http://127.0.0.1:8000`. <br/>
You can access the **interactive API documentation** at `http://127.0.0.1:8000/docs`. <br/>
If you need a cleaner read-only version, you can use `http://127.0.0.1:8000/redoc`.

## Parameters

### Path Parameters

<div class="grid" markdown>

FastAPI automatically validates, converts and binds the parameters in the path <br/>
-> `item_id` is automatically converted to an `int` and can be used as such in the function body.

```python
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

</div>

!!! warning "Order matters!"
    <div class="grid" markdown>

    <div>

    Fixed paths have to be declared **before** variable paths. Otherwise, the fixed path will never be reached. <br/>
    Also: values can be fixed with a `Enum` type, which will also be validated automatically:

    ```python
    from enum import Enum

    class ModelName(str, Enum):
        alexnet = "alexnet"
        resnet = "resnet"
    ```

    </div>

    
    ```python
    @app.get("/items/fixed") # <- fixed first
    def read_fixed():
        return {"message": "fixed path"}
    
    @app.get("/items/{item_id}")
    def read_item(item_id: int):
        return {"item_id": item_id}
    ```

    </div>

### Query Parameters (after `?`)

<div class="grid" markdown>

**Every** parameter that is not part of the path is automatically interpreted as a query parameter. <br/>
`GET /items/?skip=20&limit=5`

```python
@app.get("/items/")
def list_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

</div>

#### Query Validation & Metadata

You can use the `Query` class to add validation and metadata to your query parameters. <br/>

```python
from fastapi import Query

@app.get("/items/")
def list_names(limit: int = Query(max_length=10, description="Limit the num of items returned")):
    return {"limit": limit}
```

The most important Query parameters are:

- ^^**`max_length`/`min_length`**^^: *(for strings)* validate the length
- ^^**`ge`/`le`**^^: *(for numbers)* validate the minimum and maximum values
- ^^**`description`**^^: provide metadata for the API documentation
- ^^**`alias`**^^: use a different name for the query param than the function argument (nice when using non-legal python names like `item-id`)
- ^^**`examples`**^^: provide an example value for the API documentation -> `examples=["example1", "example2"]`
- ^^**`regex`**^^: *(only for strings)* validate the value with a regex pattern
- ^^**`deprecated`**^^: mark the query parameter as deprecated in the API documentation

If you want to type hint, validate and add default values to a query parameter, **without typecheckers complaining**, you can use the `Annotated` type from `typing`:

```python
from typing import Annotated

@app.get("/items/")
def list_names(limit: Annotated[int, Query(max_length=10, description="desrc text")]):
    return {"limit": limit}
```

## Request Body (POST)

<div class="grid" markdown>

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float

@app.post("/items/")
def create_item(item: Item):
    return item # returns to client
```

`Items` is a Pydantic model; FastAPI will automatically validate the request body against this model. <br/>
`Pydantic` also provides types like `EmailStr`, `HttpUrl`, `IPvAnyAddress`, `Field`, etc. [see docs](https://pydantic.dev/docs/validation/2.1/usage/types/standard_types/) <br/>
Nested models are also supported! <br/>
-> You stop writing validation code and can focus on your business logic.

</div>

## Response Models & status Codes

### Response Models

<div class="grid" markdown>

With `response_model` you can set a model that is allowed to go in or go out. FastAPI automatically validates the data against it and *removes not allowed values automatically*.
**Even if you return the full object, FastAPI filters it down to UserOut**

```python
class UserIn(BaseModel):
    username: str
    password: str   # comes in...

class UserOut(BaseModel):
    username: str   # ...but only this goes out

@app.post("/users/", response_model=UserOut)
def create_user(user: UserIn):
    return user
```

</div>

### Status Codes

<div class="grid" markdown>

```python
from fastapi import status

@app.post("/items/", status_code=status.HTTP_201_CREATED)
def create_item(item: Item):
    if some_error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{item} could not be created.""
        )
    return item
```

With the `Status Codes` you can more easily tell the user/browser what happened in the backend. Use the `status.HTTP_*`constants instead of raw numbers! -> reusable and autocomplete.
If you raise a Error, than you get a HTTP status code and the `detail` is the delievered payload.

</div>

#### Most important `status.HTTP_*`

**Success Codes (2xx)**

- **`status.HTTP_200_OK`**: default for standard successful request. (`GET`, `PUT`, `PATCH`)
- **`status.HTTP_201_CREATED`**:
- **`status.HTTP_204_NO_CONTENT`**: for successful request with no content to return. (`DELETE`)

**Client Error Codes (4xx):**

- **`status.HTTP_400_BAD_REQUEST`**: invalid request, e.g. missing required parameters
- **`status.HTTP_401_UNAUTHORIZED`**: authentication required, e.g. missing or invalid token
- **`status.HTTP_403_FORBIDDEN`**: authentication succeeded but the authenticated user does not have permission to access the resource
- **`status.HTTP_404_NOT_FOUND`**: resource not found
- **`status.HTTP_409_CONFLICT`**: request could not be completed due to a conflict with the current state of the resource, e.g. duplicate entry
- **`status.HTTP_422_UNPROCESSABLE_ENTITY`**: Data validation error, e.g. invalid data type or missing required fields

**Server Error Codes (5xx):**

- **`status.HTTP_500_INTERNAL_SERVER_ERROR`**: Something went wrong on the server side, e.g. unhandled exception

## `async def` vs `def`

`async def` runs directly in the event loop and is non-blocking. <br/>
`def` runs in a threadpool so it does not block the main event loop but a *worker thread* is used for the request. <br/>
BUT when using `async def`, you HAVE to use `await` for all blocking calls (like database queries, file I/O, etc.) otherwise the event loop will be blocked and the server will not be able to handle other requests. <br/>

And since a Thread is more expensive than a coroutine, you should use `async def` whenever possible but `def`is a good default if you are unsure. <br/>

<div class="grid" markdown>

```python title="async def -> concurrent"
@app.get("/items/")
async def read_items():
    # await is required for blocking calls
    items = await get_items_from_db()
    # now until query is finished, 
    # the event loop can handle other requests
    return items
```

```python title="def -> parallel"
@app.get("/items/")
def read_items():
    items = get_items_from_db()
    # this runs in a different thread
    # during that, the main event loop can handle other requests
    return items
```

</div>

## Dependency Injection

!!! abstract "Long Story Short"
    Dependency Injection is a design pattern that **prevents hard-coded** parameters by allowing you to inject them at runtime with a **"shared" dictionary** using the `Depends` class. <br/>

<div class="grid" markdown>

```python
from typing import Annotated
from fastapi import FastAPI, Depends

app = FastAPI()

def common_params(skip: int = 0, limit: int = 100):
    return {"skip": skip, "limit": limit}

CommonParams = Annotated[dict, Depends(common_params)]

@app.get("/items/")
def read_items(commons: CommonParams):
    return commons

@app.get("/users/")
def read_users(commons: CommonParams):
    return commons
```

This tells FastAPI to call the dependency function and pass its return value into the arguments of the function.<br/><br/>
To not also repeat the same code for the `Depends` in every function, we can annotate the function with `Annotated` and use it as a type hint. <br/><br/>
With `Depends` we can also verify authentication, check permissions, connect to a database, etc. since it is called *before* the actual function is executed. <br/>

</div>

## FastAPI - Project Structure

DON'T put all your code in one file! <br/>
Split routes into modules with `APIRouter` and use `include_router` to include them in the main application. <br/>

```text title="example project structure"
myproject/
├── app/
│   ├── main.py           # creates FastAPI(), includes routers, CORS
│   ├── routers/
│   │   ├── items.py
│   │   └── users.py
│   ├── models.py         # Pydantic schemas (In/Out models)
│   └── database.py       # DB session, get_db dependency
└── pyproject.toml
```

<div class="grid" markdown>

```python title="app/routers/items.py"
from fastapi import APIRouter

router = APIRouter(prefix="/items", tags=["items"])

@router.get("/")  # actual path becomes /items/
def list_items():
    return [...]

@router.get("/{item_id}") # -> /items/{item_id}
def read_item(item_id: int):
    return {...}
```

```python title="app/main.py"
from fastapi import FastAPI
from app.routers import items, users

app = FastAPI(title="My Wiki API")

app.include_router(items.router)
app.include_router(users.router)
```

</div>

- `prefix`: every route in the router will be prefixed with this path (e.g. `/items` in the example above)
- `tags`: used for grouping routes in the API documentation (e.g. `items` in the example above)

## CORS (Cross-Origin Resource Sharing)

<div class="grid" markdown>

Usually, your frontend and backend are running on different domains (e.g. `localhost:3000` for React and `localhost:8000` for FastAPI). <br/>
Browsers block requests from different origins by default. (As they should!) <br/>
To allow your frontend to access your backend, you need to enable CORS in your FastAPI application. <br/><br/>
You should **never** allow all origins in production! (`allow_origins=["*"]`) and most Browsers will block it anyway. <br/>

```python
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
    "https://myfrontend.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allow these origins
    allow_credentials=True,
    allow_methods=["*"], # allow all methods (GET, POST, etc.)
    allow_headers=["*"], # allow all headers
)
```

</div>

## Everything together

```python title="Example FastAPI application"
from typing import Annotated # for type hinting with Query, Path, etc.
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware # for your Vue frontend
from pydantic import BaseModel, Field # for data validation

app = FastAPI(title="API")

# CORS (for your Vue frontend)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Pydantic model = data contract
class Item(BaseModel):
    name: str
    price: float = Field(gt=0)

# path param + query param
@app.get("/items/{item_id}")
def get_item(item_id: int, q: str | None = None):
    if item_id > 100:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    return {"item_id": item_id, "q": q}

# request body + response model + status code
@app.post("/items/", response_model=Item, status_code=status.HTTP_201_CREATED)
def create_item(item: Item):
    return item
```

### mental Model when writing a Endpoint

1. What **method + path**? (`@app.get`, `@router.post`, ...)
2. What **comes in**? -> path params (`int`), query params (defaults), body (Pydantic model).
3. What **goes out**? -> set `response_model` to control/hide fields.
4. What **can go wrong**? -> `raise HTTPException(...)` for business errors (validation is automatic).
5. Does it need a **shared resource** (DB, current user)? -> `Depends(...)`.
6. Am I **awaiting** anything? -> `async def`. Otherwise plain `def`.
