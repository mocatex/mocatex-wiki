---
title: "Swift Basics"
icon: simple/swift
tags:
  - swift
  - programming
  - basics
---

# The Swift Programming Language

![Swift Logo](../assets/swift-icon.png){ width=20% .center}

On this site we will learn the basics of the Swift programming language. (assuming some decent knowledge of other programming languages)

## Setup Playground

For just Swift programming, in XCode, you can create a Playground:
In the start screen of XCode got to `File` > `New` > `Playground...` or ++alt+shift+command+n++ to create a `.playground` file.

## Absolute Basics

We can comment code with `//` for single line comments and `/* */` for multi-line comments. <br/>
With `// MARK: ` we can add a mark to our code to make it easier to navigate in the code navigator.

### Variables and Constants

<div class="grid" markdown>

The naming conventions for variables is `camelCase` and for constants is `PascalCase`.
We can initialize a variable/constant with a value like this:

```swift
var myVariable: Int = 10_000 // variable
let MyConstant: String = "Hey World" // constant
```

</div>

Other often used types are `Double`, `Float`, `Bool`, `Character` and `Array`.

> Multi-Line Strings are created with three double quotes `"""`

!!! info ""
    To **embed** a variable/constant into a string we can use `\(myVariable)`.

### Operators and Boolean Logic

Math operations are pretty standard, we can use `+`, `-`, `*`, `/` and `%` for addition, subtraction, multiplication, division and modulo.

Boolean expressions can be created with `==`, `!=`, `<`, `>`, `<=` and `>=` for equal, not equal, less than, greater than, less than or equal to and greater than or equal to.

We can **concatenate** strings with `+`.

### Complex Data Types

<div class="grid cards" markdown>
- ^^**Array:**^^ Lists of values of the same type. Can be created with `[]`:
```swift
var myArray: [String] = ["Hello", "World"]
```
- ^^**Set:**^^ Unordered collection of unique values of the same type. Can be created with `Set()`:
```swift
var mySet: Set<Int> = Set([1, 2, 3])
```
- ^^**Dictionary:**^^ Key-Value pairs of values of the same type. Can be created with `[:]`:
```swift
var myDictionary: [String: Int] = ["One": 1, "Two": 2]
print(myDictionary["One"], default: 0) // prints 1
```
- ^^**Enum:**^^ A type that defines a group of related values. Can be created with `enum`:
```swift
enum Direction {
    case north
    case south
    case east
    case west
}
var myDirection: Direction = .north
```
</div>

## Control Structures and Loops

### Control Structures (if/else, switch)
