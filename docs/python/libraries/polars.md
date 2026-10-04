---
title: "Polars"
icon: simple/polars
---

# Polars

![Polars Logo](../assets/polars-logo.avif)

## Basics

### Creating, Inspecting, Loading and Saving Data

<div class="grid" markdown>

```python title="Creating basic DataFrame"
import polars as pl

df = pl.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
})
```

Polars (like Pandas) works with *Series* (~ list) and *DataFrames* (~ table), where every column is a Series. <br><br>
Different from Pandas, Polars is very strict about data types -> no mixing, except you set `strict=False` or `dtype=object` when creating a DataFrame. (code-smell!)

```python title="Inspecting DataFrame"
df.head(2)  # first 2 rows
df.tail(5)  # last 5 rows
df.shape  # (rows, columns)
df.columns  # list of column names
df.describe()  # statistics for numeric col
df.glimpse()  # quick overview of the DataFrame
```

```python title="Loading and Saving DataFrame"
df.write_csv("data.csv")
df.write_parquet("data.parquet")
df2 = pl.read_csv("data.csv")
df3 = pl.read_parquet("data.parquet")
```

</div>

#### Data Types and Casting

Every column has exactly one dtype. Polars does not convert silently, you cast.

Most common dtypes: <br>
`pl.Int64`, `pl.Float64`, `pl.Utf8` (string), `pl.Boolean`, `pl.Date`, `pl.Datetime`, `pl.List`, `pl.Object` (generic Python object)

```python title="Casting Data Types"
df = df.with_columns(pl.col("age").cast(pl.Float64))  # cast age to float
pl.col("x").cast(pl.Int8, strict=False)    # failed casts become null instead of raising
```

### Polars Expressions

> Polars works with *expressions* to perform operations on DataFrames. Expressions are lazy and can be combined to create complex transformations.

```python title="Using Expressions"
add_5 = pl.col("age") + 5   # this is df agnostic -> works on any df with an 'age' column
df = df.with_columns(add_5.alias("age_plus_5"))
```

-> `.with_columns()` is used to add new columns to a DataFrame. <br>
-> `.alias()` is used to rename the resulting column.

### Selecting Data

```python title="Selecting Data"
df.select(["name", "age"])  # select specific columns
df.filter(pl.col("age") > 30)  # filter rows based on a condition -> can also be regex based
df.sort("age", reverse=True)  # sort by age in descending order
df.group_by("name").agg(pl.mean("age"))  # group by name and calculate mean
```

-> You can chain `.select()`, `.filter()`, `.sort()`, and `.group_by()` to perform multiple operations in a single expression.

??? info "Difference between `select` and `with_columns`"
    - `.select()` returns a new DataFrame with only the selected columns.
    - `.with_columns()` adds new columns to the existing DataFrame.

#### Chaining Filters

If you chain mutliple filters with a `,` (comma), it will be treated as an AND operation. <br>
You can also do it explicitly with `&` (and) or `|` (or) operators.

**Every filter must be put in parentheses `()` to avoid operator precedence issues.**

```python
df.filter((pl.col("age") > 30) & (pl.col("name").str.contains("a")))  # AND operation
df.filter((pl.col("age") > 30) | (pl.col("name").str.contains("a")))  # OR operation
```

-> Shortcut for **between**: `pl.col("age").is_between(30, 40)` <br>

### Transform Tables

#### Adding/Removing/Renaming Columns

```python title="Adding and Removing Columns"
df = df.with_columns((pl.col("age") * 2).alias("age_times_2"))
df = df.drop("age_times_2")  # remove column
df = df.rename({"age": "age_new"})  # rename column
```

-> Without `.alias()`, the new column will overwrite the existing column with the same name.

#### GroupBy and Aggregations

Mental Model: GroupBy puts the DataFrame into "buckets" based on the unique values of the specified column(s). Then, you can apply aggregation functions to each bucket on selected other columns. (Like sum them up, or take the mean, etc.)

```python title="GroupBy and Aggregations"
df.group_by("name").agg(                 # All entries with the same name are grouped together
    pl.mean("age").alias("mean_age"),   # ages in every group are averaged
    pl.sum("age").alias("sum_age"),     # ages in every group are summed (independent of other agg!)
)
```

#### `over`

Is an alternative to `group_by` and preserves the original Rows and adds the aggregated values as new columns to each row. <br>

```python
df.with_columns(
    pl.mean("age").over("name").alias("mean_age")  # mean age for each name, but keep all rows
)
```

#### Joins

<div class="grid" markdown>

Mental Model: Merging two DataFrames based on a common column (`on=`) (like SQL JOIN). Polars supports different types of joins: inner, left, outer, and cross.

```python title="Joins Structure"
df.join(df2, on="name", how="inner")
df.join(df2, left_on="id", right_on="user_id")  # different key names
```

</div>

Options for `how=`:

- `inner`: only rows with matching keys in both DataFrames; everything else is dropped
- `left`/`right`: all rows from the left/right DataFrame are kept; non-matching rows from the other DataFrame are filled with null
- `full`: all rows from both DataFrames; non-matching rows are filled with null
- `cross`: all rows from the first DataFrame are combined with all rows from the second DataFrame

#### Concatenation

This is simply stacking two DataFrames on top of each other (like SQL UNION). <br>
For that they must have the same columns (names and types). <br>

```python title="Concatenation"
df1 = pl.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
df2 = pl.DataFrame({"name": ["Charlie", "David"], "age": [35, 40]})
pl.concat([df1, df2], how="vertical")    # default, same columns
pl.concat([df1, df2], how="diagonal")    # different columns -> missing ones become null
pl.concat([df1, df2], how="horizontal")  # side by side
```

### Working with Strings and Dates

[TODO when you hit this in a project]

## Advanced Features

### Conditional Expressions

<div class="grid" markdown>

```python
df = df.with_columns(
    pl.when(pl.col("age") > 30)
    .then(pl.lit("old"))
    .otherwise(pl.lit("young"))
    .alias("age_group")
)
```

Polars allows you to create new columns based on conditions using the `when-then-otherwise` syntax. <br>
So it's similar to the Python *if/else* statement.<br>
**But:** without `pl.lit(...)` it takes the string as a column name (so used for condidtional column selection).

</div>

### Working with `Null` and `NaN` Values

??? info "`Null` vs `NaN`"
    - `Null` represents missing or undefined data.
    - `NaN` (Not a Number) is a special floating-point value that represents undefined or unrepresentable numerical results (like 0/0).

#### Handling Nulls/NaNs

<div class="grid" markdown>

```python title="Detecting rows with Null"
df.filter(pl.col("age").is_null())
df.filter(pl.col("age").is_not_null())
```

-> `.is_null()`/`.is_nan()` returns a boolean Series indicating which values are null/NaN.

```python title="Removing Null"
df.drop_nulls()  # remove rows with ANY null values
df.drop_nulls(subset=["age", "name"]) # only in specific columns
```

```python title="Filling Null"
df.fill_null(0)  # fill null values with 0
pl.col("name").fill_null("unknown") # only specific column
```

</div>

-> Same functions exist for `NaN` values: `.is_nan()`, `.drop_nans()`, `.fill_nan()`.

### Lazy API (very useful)

Polars by default uses eager execution: Every operation is executed immediately and returns a new DataFrame. <br>
The Lazy API allows you to build a query plan without executing it immediately. This then will be optimized and executed when you call `.collect()`.

You usually first build a *lazy query* and then execute or explain it afterwards.

```python title="Lazy API"
query = (
    pl.scan_csv("data.csv")  # lazy read
    .select(["name", "age"])
    .filter(pl.col("age") > 30)
)

query.collect()  # execute query and return a DataFrame
query.explain()  # show optimized query plan
```

when you already have a DataFrame (so no scan), you can convert it to lazy with `.lazy()`: `df_lazy = df.lazy()`

### Streaming API

<div class="grid" markdown>

```python
query = (pl.scan_csv("large_data.csv")
    ...
)
query.collect(engine="streaming")
```

-> Loads data in chunks and processes it in a streaming fashion, which prevents memory overflow for large datasets.

</div>
