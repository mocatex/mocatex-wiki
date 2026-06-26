---
title: Recursion
icon: lucide/rotate-cw
---

# Recursion

Quick Info: Recursion is, where a function calls itself in order to solve a problem.

## Primitive Recursion

The most basic form of recursion is when a function calls itself with a smaller input until it reaches a base case. This is often used to solve problems that can be broken down into smaller subproblems.

```haskell title="Example of primitive recursion"
factorial :: Integer -> Integer
factorial 0 = 1 -- base case
factorial n = n * factorial (n - 1) -- recursive case
```

## Value Recursion

In value recursion, a function calls itself with the same input, but it uses the result of the recursive call to compute the final result. This is often used to solve problems that require multiple passes over the data.

```haskell title="Example of value recursion"
fibonacci :: Integer -> Integer
fibonacci 0 = 0 -- base case
fibonacci 1 = 1 -- base case
fibonacci n = fibonacci (n - 1) + fibonacci (n - 2)
```

## Tail Recursion

Tail recursion is a special form of recursion where the recursive call is the last operation performed in the function. This allows the compiler to optimize the recursive calls and avoid stack overflow for large inputs.

!!! warning "What is actually tail recursive"

    Not everything that looks like tail recursion is actually tail recursive. For example:
    
    ```haskell title="Not actually tail recursive"
    sumList :: [Integer] -> Integer
    sumList [] = 0 -- base case
    sumList (x:xs) = x + sumList xs
    ```
    not tail recursive because we have to wait for the result of sumList xs before we can add x

To make it tail recursive, we can use either the "accumulator" pattern or the "continuation" pattern.

### Accumulator Pattern

Instead of waiting for the result of the recursive call, we can pass an additional parameter (the accumulator) that keeps track of the intermediate result.

```haskell title="Tail recursive version using the accumulator pattern"
sumList :: Integer -> [Integer] -> Integer
sumList acc [] = acc -- base case
sumList acc (x:xs) = sumList (acc + x) xs -- recursive case
```

But not all problems can be solved with the accumulator pattern. For example, the Fibonacci sequence cannot be computed in a tail recursive way using the accumulator pattern. However, we can use the continuation pattern to solve this problem.

### Continuation Pattern

In the continuation pattern, we pass an additional parameter (the continuation) that represents the rest of the computation that needs to be performed after the recursive call.

```haskell title="Tail recursive version using the continuation pattern"
sumList :: [Integer] -> (Integer -> Integer) -> Integer
sumList [] cont = cont 0 -- base case
sumList (x:xs) cont = sumList xs (\result -> cont (x + result)) -- recursive case
```

## Fixpoints

A fixpoint of a function is a value that is unchanged by the function. In Haskell, we can use the `fix` function from the `Data.Function` module to compute the fixpoint of a function.

```haskell title="Using fix to compute the fixpoint of a function"
import Data.Function (fix)

factorial :: Integer -> Integer
factorial = fix $ \f n -> if n == 0 then 1 else n * f (n - 1)
```

What happens here in easy terms: `fix` takes a function (here a lambda function that takes `f` and `n`) and returns a new function that is the fixpoint of that function. The lambda function is defined in terms of itself (`f`), which allows us to achieve recursion without naming the function.