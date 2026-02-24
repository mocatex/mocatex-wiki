---
title: Functions
icon: material/function-variant
---

# haskell - Functions

TODO: transform prefix to infix AND declare native infix functions

## Function Basics

<div class="grid" markdown>
<div>
The general syntax for defining basic functions is:

```haskell
functionName :: InputType -> OutputType
functionName input = expression
```

</div>

```haskell title="Example: add 5 to a number"
addFive :: Integer -> Integer
addFive x = x + 5
```

functions can also be writen as **lambda functions** (also: anonymous functions) using the `\` symbol and `->` instead of the `=` sign.
^^Note:^^ in this example the lamda function is kind of redundant since we assign it to a name. You would usually use them as arguments, like in the `map` function.

```haskell title="Example: add 5, using a lambda function"
addFiveLambda :: Integer -> Integer
addFiveLambda = \x -> x + 5

-- example of actual usage:
numbers = map (\x -> x + 5) [2, 3] > [7, 8]
```

</div>

> You use the function by writing its ***name followed by the argument(s)*** separated by spaces. For example, `addFive 10` would return `15`.

## Functions with Multiple Arguments

### Arguments as Tuples

The easiest way to define a function with multiple arguments is to use tuples. But it's also the least flexible way, since you have to pass all arguments at once as a tuple.

```haskell
add :: (Integer, Integer) -> Integer
add (x, y) = x + y
```

### Curried Functions

Curried functions are a powerful feature of Haskell which allow you to partially apply functions. Let's again use the `add` function as an example:

```haskell
add :: Integer -> Integer -> Integer
add x y = x + y
```

We have now three `Integer` types in the type signature. The trick in reading this is: ^^**group from the right**^^. So we can rewrite this as:

```haskell
add :: Integer -> (Integer -> Integer)
```

Now we can read this like in a recursive way: `add` takes an `Integer` and returns a function that takes another `Integer` and returns an `Integer`. 
So if we disect our `add` function:

`add x y = x + y` -> `add x = \y -> x + y` -> `add = \x -> \y -> x + y`

This means we can **partially call** the `add` function by providing only one argument, which will return a new function that takes the remaining argument. For example: `add 3`will return a new function waiting for `y` to which 3 will be added.

So `add 3 4` is the same as `(add 3) 4` which is the same as `(\y -> 3 + y) 4` which finally evaluates to `7`.

Bigger and more complex functions can be curried in the same way, allowing for a lot of flexibility and code reuse:

```haskell
multiply :: Integer -> Integer -> Integer -> Integer
multiply x y z = x * y * z

-- is the same as:
multiply :: Integer -> (Integer -> (Integer -> Integer))
multiply x = \y -> \z -> x * y * z
```

??? tip "Why use curried functions?"

    There are three usecases where curried functions are especially useful:

    1. **Partial application**: As already mentioned, you can partially apply curried functions to create new functions with some arguments fixed. This allows to create "specialized" functions from more general ones.
    2. **Higher-order functions**: Curried functions can be easily passed as arguments to higher-order functions, which are functions that take other functions as arguments or return them as results.
    3. **Functional Pipelines**: (will be covered later) short: curried functions can be easily composed together to create complex operations in a clear and concise way.

!!! success "General Reading Strategy"

    If you take a chain: `a -> b -> c -> d`, you can read it like: `a -> (b -> (c -> d))`, since ^^**`->` groups from the right!**^^

    - Give Input `a` -> returns a function
    - That function takes `b` -> returns another function
    - That function takes `c` -> returns the final result `d`

## Higher-Order Functions

You can also define functions that take other ^^**functions as arguments**^^ or ^^**return them as results**^^.

```haskell title="Example: function as argument and return value"
applyTwice :: (a -> a) -> a -> a
applyTwice f x = f (f x)
```

Here `applyTwice` takes a function `f` that takes an argument of type `a` and returns a value of the same type `a`, and also takes an argument `x` of type `a`. It applies the function `f` to `x` twice, effectively returning `f (f x)`.

> You can also write this using lambda functions:
> 
> ```haskell
> applyTwice :: (a -> a) -> a -> a
> applyTwice = \f -> \x -> f (f x)
> ```
>
> or even more concisely using **function composition**:
> 
> ```haskell
> applyTwice :: (a -> a) -> a -> a
> applyTwice f = f . f
> ```

## List Mapping

List mapping is a common usage of higher-order functions, where you apply a function to each element of a list. We now look at the most simple way of using the `map` function. (it's actually way more powerful, but we will cover that later)

```haskell title="Example: using map to add 5 to each element in a list"
numbers = [1, 2, 3, 4]
result = map (\x -> x + 5) numbers -- result will be [6, 7, 8, 9]
```

## Prefix and Infix Functions

In Haskell, functions can be used in two ways: as **prefix** or **infix**. By default, functions are used in prefix notation, where the function name comes before its arguments.
But you can use them in infix notation by surrounding the function name with backticks (`` ` ``). This allows you to write expressions in a more natural way, especially for binary functions.

```haskell title="Example: using infix notation"
add :: Integer -> Integer -> Integer
add x y = x + y

-- Using prefix notation (default)
result1 = add 3 4

-- Using infix notation
result2 = 3 `add` 4
```

Operators (`+`, `-`, `++`, `==`, etc.) are natively infix functions. Yet you can still use them in prefix notation by surrounding them with parentheses:

```haskell title="Example: using operators in prefix notation"
result3 = (+) 3 4 -- result will be 7
```

To make a custom function infix by default, you simply surround it in parentheses when defining it:

```haskell title="Example: defining an infix function"
(***) :: Integer -> Integer -> Integer
x *** y = x * y + 1 -- Now you can use it in infix notation without backticks
result4 = 3 *** 4 -- result will be 13
```