---
title: "Control Structures"
icon: lucide/git-fork
---

# haskell - Control Structures

## If-Else

> general structure: `if <condition> then <expression1> else <expression2>`

We can use `if-else` either as an expression or as a statement. When used as an expression, it returns a value based on the condition. When used as a statement, it executes different blocks of code based on the condition.

```haskell
-- if-else as an expression
result :: String
result = if x > 0 then "Positive" else "Non-positive"

-- if-else as a statement
if x > 0 then
    putStrLn "Positive"
else
    putStrLn "Non-positive"
```

You can also use multiple **nested** `if-else` statements to handle more complex conditions:

```haskell
result :: String
result = if x > 0 then "Positive"
         else if x < 0 then "Negative"
         else "Zero"
```

### Guards

Yet, for more complex conditions, it is often better to use guards.
The general syntax for guards is:

```haskell
functionName args
    | condition1 = expression1
    | condition2 = expression2
    | otherwise  = defaultExpression
```

First condition that evaluates to `True` will determine the result and the rest will be ignored. `otherwise` is a catch-all condition that will be used if none of the previous conditions are `True`.
You can also combine multiple conditions with `&&` and `||`

```haskell title="Using guards with multiple conditions"
result :: Integer -> String
result x
    | x % 3 == 0 && x % 5 == 0 = "FizzBuzz"
    | x % 3 == 0 = "Fizz"
    | x % 5 == 0 = "Buzz"
    | otherwise  = show x
```

## Cases and Pattern Matching

### Case Expressions

Case expressions are like switch statements in other languages. They allow you to match a value against multiple patterns and execute different code based on the match.

We can write them in two different ways: 

<div class="grid" markdown>

```haskell title="multiple clauses"
scrollDirection :: Integer -> String
scrollDirection 1 = "Up"
scrollDirection (-1) = "Down"
scrollDirection _ = "No Scroll"
```

```haskell title="case expression"
scrollDirection :: Integer -> String
scrollDirection x = case x of
    1 -> "Up"
    (-1) -> "Down"
    _ -> "No Scroll"
```

</div>

### Pattern Matching

Here the true power of case expressions comes into play. We can use them together with custom data types to match on specific patterns.

```haskell title="pattern matching with custom data types"

data Shape = Circle Float 
    | Rectangle Float Float

area :: Shape -> Float
area shape = case shape of
    Circle r -> pi * r^2
    Rectangle w h -> w * h
```

## Let and Where

Both `let` and `where` are used to define local variables and functions within a larger expression.

### Let Bindings

We use `let` to define **local variables** within an expression. So they are only accessible within that scope.

The general syntax for `let` is:

```haskell
let <bindings> in <expression>
```

Here `<bindings>` is where you define your local variables and functions, and `<expression>` is where you can use those bindings.

```haskell title="Example: let to define local variables"
isBigCircle :: Double -> Bool
isBigCircle r =
    let area = pi * r^2
    in area > 50
```

### Where Clauses

`where` is similar to `let`, but it is used to define local variables and functions **AFTER** the main expression. The variables defined in a `where` clause are also only accessible within that scope.

> Rule: You can always use `let` instead of `where`, but not the other way around.

```haskell title="Example: where to define local variables"
isBigCircle :: Double -> Bool
isBigCircle r = area > 50
    where area = pi * r^2
```
