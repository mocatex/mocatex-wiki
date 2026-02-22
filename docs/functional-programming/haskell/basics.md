---
title: Basics
icon: lucide/notebook-pen
---

# Haskell - the basics

In the last chapter we set up our Haskell environment and saw how we can run Haskell code. In this chapter we will cover the basics of the Haskell language.

Theoretically every full haskell program should have a `main` function, but for the sake of learning we will skip this for now and just write our functions and expressions in a `.hs` file and load and run them in GHCi (REPL).

## Basic Syntax

<div class="grid" markdown>

^^**Comments**^^: Haskell supports single-line comments using `--` and multi-line comments using `{- -}`.

```haskell
-- This is a single-line comment
{- This is a
multi-line comment -}
```

^^**Variables**^^: we need to define the type of the variable first, then we can assign a value to it. 

```haskell
x :: Integer
x = 5
```

</div>

## Numbers and Operations

<div class="grid" markdown>

Haskell has all the basic arithmetic operations: (`+`, `-`, `*`, `/`) and also supports more advanced operations like exponentiation (`^`) and modulus (`mod`) or integer division (`div`).

```haskell
x = 5 + 4 - 3 * 2 / 1
y = 2 ^ 3
z = 10 `mod` 3
w = 10 `div` 3
```

</div>

> For why we use backticks for `mod` and `div` see the chapter on [infix vs prefix functions](../functions.md#infix-vs-prefix-functions).

## Booleans and Logic

<div class="grid" markdown>

Haskell has a `Bool` type which can be either `True` or `False`. We can use logical operators like `&&` (and), `||` (or) and `not` (negation).

```haskell
a = True && False -- > False
b = True || False -- > True
c = not True -- > False
```

</div>

> Booleans also come into play when we use comparison operators:
> 
> - `==` & `/=` for equality and inequality
> - `<`, `>`, `<=`, `>=` for ordering

## Strings and Characters

<div class="grid" markdown>

Strings are defined using **double quotes** `"` and characters are defined using **single quotes** `'`. You can ^^concatenate^^ strings using the `++` operator or uring the `concat` function.

```haskell
str1 = "World" :: String
char1 = 'A' :: Char
str2 = "Hello " ++ str1 -- > "Hello World"
str3 = concat ["Hi ", "World"] -- > "Hi World"
```

</div>

## Lists

<div class="grid" markdown>

Lists in Haskell are usually *linked lists* and must be **homogeneous** (all elements must be of the same type). We define them using square brackets `[]` and commas to separate the elements.

```haskell
myList :: [Integer]
myList = [1, 2, 3, 4, 5]
nestedList :: [[Integer]]
nestedList = [[1, 2], [3, 4], [5]]
```

</div>

### List Comprehensions

We can create new lists from existing ones using list comprehensions, ranges and filters.

<div class="grid" markdown>
<div>
The general syntax for list comprehensions is:

```haskell
[ expression | element <- list, condition ]
```

Where <code>expression</code> is the value we want to generate, <code>element</code> is a variable that takes on each value from <code>list</code> (also: generator), and <code>condition</code> is an optional filter that determines which elements are included.

</div>

!!! example Full Example

    Create a list of ^^squares^^ of ^^even^^ numbers ^^from 1 to 10^^:

    - square -> expression
    - even numbers -> condition
    - numbers from 1 to 10 -> generator

    ```haskell
    squaresOfEven :: [Integer]
    squaresOfEven = [x^2 | x <- [1..10], x `mod` 2 == 0]
    ```

</div>

If you create tuples in a list you therefore need two (or more) generators:

```haskell
tupleList :: [(Integer, Integer)]
tupleList = [(x, y) | x <- [1..3], y <- [4..6]]
-- > [(1,4),(1,5),(1,6),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6)]
```

And you can add **multiple conditions** (guards) by separating them with commas. **Both conditions** must be satisfied for the element to be included in the resulting list.
