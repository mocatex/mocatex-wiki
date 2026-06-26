---
title: Higher Order Functions
icon: lucide/chart-spline
---

# Higher Order Functions

> This section is a cluster of so called "higher-order functions" (HOF) that are very useful when working with lists and other data structures. They allow us to write more concise and expressive code by abstracting away common patterns of computation.

## foldl and foldr

The `foldl` and `foldr` functions are used to reduce a list to a single value by applying a binary function to the elements of the list.

`foldl` -> starts from the left, `foldr` -> starts from the right

!!! warning "always use `foldl'`"
    The standard `foldl` is not tail-recursive and therefore not recommended.
    Instead, use `foldl'` from `Data.List` which is a strict version of `foldl` and will not cause stack overflow for large lists.

```haskell title="foldl' composition"
foldl' function initialValue list
```

- **function**: how do I combine the "current result" with the "next element"?
- **initialValue**: the starting point for the reduction (e.g. 0 for summing, 1 for multiplying)
- **list**: the list of elements to be reduced

^^**Example**^^: `foldl' (+) 0 [1, 2, 3]` will compute `((0 + 1) + 2) + 3` and return `6`

### `foldr` - the "Outside-In" Onion

`foldr` looks at the **left** (head) first, but it doesn't "calculate" the result until it reaches the end of the list. It builds up a chain of function applications that will be evaluated once it reaches the end.

That allows it to work with infinite lists (as long as the function is lazy enough) and construct new lists from other lists.

### `foldl'` - the "Inside-Out" Snowball

`foldl'` starts with the initial value and applies the function to it and the first element of the list, then takes that result and applies the function to it and the second element, and so on. It builds up a single value as it goes through the list.

Thats mkes it very efficient for operations that can be computed in a single pass (like summing a list), but it cannot handle infinite lists and is not suitable for constructing new lists.

!!! tip "When to use which?"

    - Use `foldl'` when you want to reduce a list to a single value and the operation is associative and has an identity element (e.g. summing, multiplying).
    - Use `foldr` when you want to construct a new list from an existing list or when working with infinite lists.

## partial functions and optional values

A partial function is a function that is not defined for all possible inputs. (e.g. `div x y` is not defined for `y = 0`)

In Haskell, we can represent optional values using the `Maybe` type. It can either be `Just a` (where `a` is the value) or `Nothing` (indicating the absence of a value).

```haskell title="Using Maybe to represent optional values"
safeDiv :: Int -> Int -> Maybe Int
safeDiv x y = case y of
    0 -> Nothing
    _ -> Just (x `div` y)
```
