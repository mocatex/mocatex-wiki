---
title: "CSS Selectors"
icon: lucide/square-dashed-mouse-pointer
---

# CSS Selectors

<div class="grid" markdown>

> CSS selectors are patterns used to select and style HTML elements. Selectors can be as complex as you need, allowing you to target specific elements based on their type, class, ID, attributes, and more.

```css title="Basic Structure"
selector, optional-more-selectors {
  property: value;
}
```

</div>

## Types of Selectors

We can categorize CSS selectors into several types based on their functionality and specificity. Below are some of the most commonly used selectors:

- **Universal Selector (`*`)**: Selects all elements on the page.
- **Type Selector (`element`)**: Selects all elements of a specific type (e.g., `div`, `p`, `h1`).
- **Class Selector (`.class`)**: Selects all elements with a specific class attribute.
- **ID Selector (`#id`)**: Selects the element with a specific ID attribute. (don't overuse ID's! if possible: class > id)
- **Descendant Selector (`ancestor descendant`)**: Selects all elements that are descendants of a specified ancestor element. (e.g., `div p` selects all `<p>` elements inside `<div>` elements).
- **Adjacent Sibling Selector (`element + element`)**: Selects an element that is immediately preceded by a specified sibling element. (e.g., `h1 + p` selects the first `<p>` element that follows an `<h1>`).
- **Direct Child Selector (`parent > child`)**: Selects all elements that are direct children of a specified parent element. (e.g., `ul > li` selects all `<li>` elements that are direct children of a `<ul>`).
- **Attribute Selector (`[attribute]`)**: Selects elements based on the presence or value of an attribute. (e.g., `[type="text"]` selects all input elements with type "text").
- **Pseudo-classes (`:pseudo-class`)**: Selects elements based on their ^^state^^ or position in the document. (e.g., `:hover`, `:first-child`, `:nth-child(n)` -> `a:hover {...}`).
- **Pseudo-elements (`::pseudo-element`)**: Selects and styles specific parts of an element. (e.g., `::before`, `::after`, `::first-letter`).

### Combining Selectors

We can combine selectors to create more specific rules.

<div class="grid" markdown>

Selects all `<p>` elements when hovered with the class "highlight" that are inside a `<div>`:

```css
div p.highlight:hover {
    color: red;
}
```

</div>

## Specificity

In CSS, specificity determines which styles are applied to an element when multiple selectors target the same element. The specificity hierarchy is as follows (higher specificity overrides lower specificity):

!!! note ""
    1. **Inline styles** (e.g., `style="..."`) have the highest specificity.
    2. **ID selectors** (`#id`) have higher specificity than class selectors.
    3. **Class selectors** (`.class`), **attribute selectors** (`[attribute]`), and **pseudo-classes** (`:pseudo-class`) have the same specificity level.
    4. **Type selectors** (`element`) and **pseudo-elements** (`::pseudo-element`) have the lowest specificity.

When multiple selectors have the same specificity, the last one defined in the CSS will take precedence.

There is also the `!important` rule, which can be used to override normal specificity rules. However, it should be used sparingly, as it can make debugging and maintaining CSS more difficult.

!!! info "CSS Inheritance"
    CSS properties can be inherited by child elements from their parent elements. This means that if a parent element has a certain style applied, its children will also have that style unless explicitly overridden.

