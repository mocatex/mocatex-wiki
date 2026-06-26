---
title: Functors, Applicatives, and Monads
icon: lucide/table-of-contents
---

# Functors, Applicatives, and Monads

In Haskell, functors, applicatives, and monads are powerful abstractions that allow us to work with computations in a more flexible and composable way. They are all related to the concept of mapping and chaining operations on values wrapped in a context (like a container).

## Functors

Functors are kind of a box that can hold a value and allows us to apply a function to the value inside the box without changing the structure of the box itself. The key function for functors is `fmap`, which takes a function and a functor and applies the function to the value inside the functor.

With that we can for example solve the `NullPointerException` problem in a more elegant way. Instead of checking for `null` values and handling them separately, we can use the `Maybe` functor to represent optional values and use `fmap` to apply functions to the values inside the `Maybe` functor without worrying about `null` values.

```haskell title="Using fmap with a functor"
import Data.Maybe (fromJust)

-- Maybe is a functor that can either be Just a value or Nothing
maybeValue :: Maybe Int
maybeValue = Just 5

-- Using fmap to apply a function to the value inside the Maybe functor
result :: Maybe Int
result = fmap (+1) maybeValue -- result will be Just 6
```

We can either use `fmap` directly or we can use the infix operator `<$>` which is just a synonym for `fmap`.

### Functor Laws

Functors must satisfy two laws:

1. Identity: `fmap id == id` (applying the identity function should not change the functor)
2. Composition: `fmap (f . g) == fmap f . fmap g` (applying a composition of functions should be the same as applying each function separately)

## Applicatives

Applicatives are an extension of functors that allow us to apply functions that are themselves wrapped in a context (like a functor) to values that are also wrapped in a context. The key function for applicatives is `<*>`, which takes a functor that contains a function and another functor that contains a value and applies the function to the value.

```haskell title="Using <*> with an applicative"
fmap (+) (Just 3) -- result will be Just (+3)
-- Now here we couldn't apply fmap directly because we have a function wrapped in a functor, but we can use <*> to apply it to another functor
result :: Maybe Int
result = Just (+3) <*> Just 5 -- result will be Just 8
```

## Monads

Monads are again an extension of applicatives that allow us to chain operations that are wrapped in a context. Like boxes inside boxes. The key function for monads is `>>=`, which takes a value wrapped in a monad and a function that takes a normal value and returns a value wrapped in a monad, and applies the function to the value inside the monad.

```haskell title="Using >>= with a monad"

-- Maybe is also a monad, so we can use >>= to chain operations on Maybe values
result :: Maybe Int
result = Just 5 >>= (\x -> Just (x + 1)) -- result will be Just 6
```
