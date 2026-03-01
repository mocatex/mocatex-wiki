---
title: "Types"
icon: lucide/pilcrow-right
---

# haskell - Types

We already covered the most important integrated types in Haskell in the [basics chapter](01-basics.md). In this chapter we will cover how to define **our own types** and how to use them.

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
