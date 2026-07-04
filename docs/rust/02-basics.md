---
title: "Rust Basics"
icon: lucide/list-start
---

# Rust Basics

## Variables and Mutability

<div class="grid" markdown>

In Rust by default, variables are **immutable**. We can make it mutable by using the `mut` keyword. Here are all the types of variables we can create in Rust:

```rust
let x = 5; // immutable variable
let mut y = 10; // mutable variable
const MAX_POINTS: u32 = 100_000; // constant
```

</div>

**Constants**: File scope/level, compile time ; **immutable variable**: Block scope, run time. <br/>
What we can do is **reassign** a variable  (variable shadowing) by using the same variable name again.

??? info "unused variable"
    If we create a variable and don't use it, Rust will give us a warning. We can silence this warning by prefixing the variable name with an underscore `_`. For example:

    ```rust
    let _x = 5; // unused variable
    ```

### Print Variables

We can print variables using the `println!` macro in multiple ways. Here are some examples:

```rust
let x = 5; let y = 10; let z = 15;
println!("first: {}, second: {}, third: {}", x, y, z); // {} placeholders
println!("first: {0}, second: {1}, third: {2}", x, y, z); // positional arguments
println!("first: {x}, second: {y}, third: {z}"); // named arguments
```

!!! note ""
    The `placeholder` and `positional arguments` allow also for expressions (named arguments **don't**), for example:

    ```rust
    println!("first: {}, second: {}, third: {}", x + 1, y * 2, z - 3); // expressions
    ```

## Data Types

<div class="grid" markdown>

Rust allows us to *not* specify the type of a variable (it then infers the type), but we can also specify the type explicitly. Here are some examples of specifying the type explicitly:

```rust
let x: i32 = 5; // integer
let y: f64 = 10.0; // floating point
let z: bool = true; // boolean
let s: &str = "Hello, world!"; // string slice
```

</div>

??? abstract "Most important data types"
    The most important data types in Rust are:

    - **Integer types**: `i8`, `i16`, `i32`, `i64`, `i128`, `isize` (signed) and `u8`, `u16`, `u32`, `u64`, `u128`, `usize` (unsigned)
    - **Floating point types**: `f32`, `f64`
    - **Boolean type**: `bool`
    - **Character type**: `char`
    - **String types**: `String` (owned string) and `&str` (string slice)
    - **Compound types**: `tuple` and `array`
    - **Function types**: `fn` and `closure`

### Type Aliases

<div class="grid" markdown>

Custom types can be helpful to make our code more readable. We can create a type alias using the `type` keyword. Here is an example:

```rust
type Kilometers = i32; // type alias
let x: Kilometers = 5; // using the type alias
```

</div>