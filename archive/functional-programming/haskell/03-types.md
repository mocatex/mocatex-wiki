---
title: "Types"
icon: lucide/pilcrow-right
---

# haskell - Types

We already covered the most important integrated types in Haskell in the [basics chapter](01-basics.md). In this chapter we will cover how to define **our own types** and how to use them.

## Basic Custom Types

We can create new types using either the `type` or the `newtype` keyword:

- `type` creates a type alias, which is just another name for an existing type. It does not create a new type! Can be useful for improving readability but it does not provide any type safety.

    ```haskell title="Creating a type alias"
    type Name = String -- > Name is now just another name for String
    ```

- `newtype` creates a new type that is distinct from the existing type. It is a wrapper around an existing type and provides type safety.

    ```haskell title="Creating a new type"
    newtype Name = Name String -- > Name is now a new type that wraps a String
    ```

## Product Types (Records)

> **AND** types -> a value contains multiple fields

It is somewhat similar to a data-class in Python or a struct in C. We can define a product type using the `data` keyword and then we can create instances of this type. (kind of like data-objects)

<div class="grid" markdown>

```haskell
data Person = Person
    { name :: String,
      age :: Int
    }
```

Now we our own type `Person` with two fields: `name` and `age`. When we create an instance of it: `Tom :: Person`, we HAVE to provide values for both fields. 

</div>

On the **left** side of the `=` we have the ^^name of the type^^ `Person` and on the **right** side we have the ^^constructor^^ `Person` with which we can create instances of the type. (They can have different names, but standard convention is to name them the same)

```haskell title="Creating an instance of a product type"
tom :: Person
tom =
    Person
        { name = "Tom",
          age = 30
        }
```

### Getters

We can extract the fields of a product type using the field names as functions:

```haskell title="Extracting fields from a product type"
tomName :: String
tomName = name tom -- > "Tom"   
```

> You can use the fields in general as functions. e.g. together with `map` or `filter` when you have a list of `Person`s.

### Setters

Since Haskell is an immutable language, we cannot change the fields of an existing instance. Instead, we can create a new instance with the updated fields:

```haskell title="Updating fields of a product type"
olderTom :: Person
olderTom = tom { age = 31 } -- > Person { name = "Tom", age = 31 }
```

## Sum Types (Unions)

> **OR** types -> a value can be one of multiple variants

A sum type is a type that can be one of several different types. We define it the same way as a product type, but we separate the different variants with a `|` instead of using fields.

<div class="grid" markdown>

```haskell
data paymentMethod
    = CreditCard String Int -- > card number and cvv
    | PayPal String -- > email
    | BankTransfer String String -- > account number and routing number
```

Now we have a type `PaymentMethod` that can be either a `CreditCard`, `PayPal` or `BankTransfer`. These are called the ^^constructors^^ of the sum type. Each constructor can have different fields, but they all belong to the same type `PaymentMethod`.

</div>

When we create an instance of a sum type, we have to specify which constructor we want to use:

```haskell title="Creating an instance of a sum type"
myPayment :: PaymentMethod -- > type of the sum type
myPayment = CreditCard "1234 5678 9012 3456" 123 -- > instance of the sum type
```

### Recursive Sum Types

A sum type can also be recursive, meaning that one of the constructors can contain the type itself. This is useful for defining data structures like trees or linked lists.

```haskell title="Defining a recursive sum type"
data Tree a -- > a is a type variable, it can be any type
    = Leaf a -- > end of the tree
    | Node (Tree a) (Tree a) -- > a node with two subtrees
```

## Combining Unions and Records

We can also combine product and sum types to create more complex data structures. For example, we can define a `Shape` type that can be either a `Circle` or a `Rectangle`, and each of these can have their own fields:

```haskell title="Combining product and sum types"
data Shape
    = Circle { radius :: Double }
    | Rectangle { width :: Double, height :: Double }
```

## TypeClasses

> a bit like *interfaces* in OOP languages -> a type that defines a set of functions that can be implemented by other types

Haskell has the *big three* typeclasses: `Eq`, `Ord` and `Show`:

- `Eq` defines the `==` and `/=` functions for equality comparison
- `Ord` defines the `<`, `>`, `<=` and `>=` functions for ordering comparison (must also implement `Eq`)
- `Show` defines the `show` function for converting a value to a string

!!! info adding typeclasses
    We can add typeclasses to our own types by using the `deriving` keyword:

    ```haskell title="Deriving typeclasses for a custom type"
    data Person = Person
        { name :: String,
          age :: Int
        } deriving (Eq, Ord, Show)
    ```

    Now we can compare `Person` instances for equality, order them and convert them to strings.

### Custom Typeclasses

We can also define our own typeclasses to specify a set of functions that can be implemented by other types. For example, we can define a `Describable` typeclass that requires a `describe` function:

```haskell title="Defining a custom typeclass"
class Describable a where
    describe :: a -> String
```

When we now want to make a type an instance of the `Describable` typeclass, we have to implement the `describe` function for that type:

```haskell title="Making a type an instance of a custom typeclass"
instance Describable Person where
    describe person = name person ++ " is " ++ show (age person) ++ " years old."
```

### Typeclass constraints

With typeclass constraints we can specify in the type signature of a function, what typeclasses the input types must implement.

<div class="grid" markdown>

```haskell title="Using typeclass constraints in a function"
describeList :: (Describable a) => [a] -> [String]
describeList xs = map describe xs
```

Here `describeList` takes a list of any type `a`, as long as `a` is an instance of the `Describable` typeclass. This allows us to use the `describe` function on the elements of the list, even though we don't know their specific types.

</div>

If we have multiple typeclass constraints, we can separate them with a comma.
