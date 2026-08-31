---
title: "CSS Flexbox"
icon: lucide/layout
---

# Responsive CSS & Flexbox

> Flexbox is a extremely powerful layout system in CSS that allows us to create flexible and responsive layouts with ease.

You can add this to a element by using the `display: flex;` property. This will make the element a **flex container**, and its **children** will become **flex items**.

!!! info "Flexbox Directions"
    Flexbox has two main axes:
    - **Main Axis**: The primary axis along which flex items are laid out. Default is horizontal (row), but can be changed to vertical (column) using the `flex-direction: column;` property.
    - **Cross Axis**: The axis perpendicular to the main axis. If the main axis is horizontal, the cross axis is vertical, and vice versa.
    - If you need a *reverse* direction, you can use `flex-direction: row-reverse;` or `flex-direction: column-reverse;`.

## Flex Container Properties

### Justify Content

> The `justify-content` property aligns flex items **along the main axis**. It can take the following values:

<div class="grid" markdown>

- `flex-start`/`flex-end`: Aligns items to the start or end of the container.
- `center`: Centers items along the main axis.
- `space-between`: Distributes items evenly, with the first item at the start and the last item at the end.
- `space-around`: Distributes items evenly with equal space around them. (Space between the items is twice as much as the space at the edges.)
- `space-evenly`: Distributes items evenly with equal space between them. (Every Space is the same, including the edges.)
- `stretch`: Stretches items to fill the container (default behavior for flex items).

![justify-content-illustration](../assets/justify-content.avif)

</div>

### Flex Wrap

> The `flex-wrap` property controls whether flex items should wrap onto multiple lines or stay on a single line. It can take the following values

- `nowrap`: All flex items will be on a single line (default).
- `wrap`: Flex items will wrap onto multiple lines, from top to bottom.
- `wrap-reverse`: Flex items will wrap onto multiple lines, but in reverse order (from bottom to top).

### Align Items

> The `align-items` property aligns flex items **along the cross axis**. It can take the following values:

- `flex-start`/`flex-end`: Aligns items to the start or end of the cross axis.
- `center`: Centers items along the cross axis.
- `baseline`: Aligns items along their baseline. (This is useful for text alignment.)

### Align Content

> The `align-content` property aligns flex lines **along the cross axis** when there are multiple lines of flex items. So for example when there are rows of items, this property will align and manage the spacing between those rows. It can take the following values:

<div class="grid" markdown>

- `flex-start`/`flex-end`: Aligns lines to the start or end of the cross axis.
- `center`: Centers lines along the cross axis.
- `space-between`: Distributes lines evenly, with the first line at the start and the last line at the end.
- `space-around`: Distributes lines evenly with equal space around them
- `space-evenly`: Distributes lines evenly with equal space between them.

![align-content-illustration](../assets/align-content.avif)

</div>

-> so you see the options start to repeat themselves! The idea behind it is the same.

### Align Self

> The `align-self` property allows you to override the `align-items` property for individual flex items. It can take the same values as [`align-items`](#align-items)!

## Flex Item Properties

### Flex-Basis

The `flex-basis` property defines the initial height or width of a flex item before it is distributed in the flex container. It's along the main axis, so:
`flex-direction: row;` -> width, `flex-direction: column;` -> height.
Any width/height set on the flex item will be overridden by `flex-basis`!

You can set it to a specific value (like `200px`), or use `auto` (default) to let the item size itself based on its content.

### Flex-Grow

The `flex-grow` property defines how much a flex item should **grow relative to the other items** in the flex container. (Not how much bigger it gets than the others.) It takes a unitless value, which acts as a proportion. For example:

- If all items have `flex-grow: 1;`, they will grow equally to fill the available space.
- If one item has `flex-grow: 2;` and the others have `flex-grow: 1;`, the item with `flex-grow: 2;` will take up twice as much space as the others.
- If an item has `flex-grow: 0;`, it will not grow at all, even if there is available space.

You can protect them from getting too big or too small by using `min-width`/`max-width` or `min-height`/`max-height`.

### Flex-Shrink

The `flex-shrink` property defines how much a flex item should **shrink relative to the other items** in the flex container when there is not enough space. It also takes a unitless value, which acts as a proportion. For example:

- If all items have `flex-shrink: 1;`, they will shrink equally when the container is too small.
- If one item has `flex-shrink: 2;` and the others have `flex-shrink: 1;`, the item with `flex-shrink: 2;` will shrink twice as much as the others.
- If an item has `flex-shrink: 0;`, it will not shrink at all, even if there is not enough space.

So it works the same way as `flex-grow`, but in reverse.

### Flex Shorthand

!!! success "Flex Shorthand"
    The `flex` property is a shorthand for `flex-grow`, `flex-shrink`, and `flex-basis`. It can take one, two, or three values:

    - One value: `flex: 1;` (equivalent to `flex-grow: 1; flex-shrink: 1; flex-basis: 0%;`)
    - Two values: `flex: 2 1;` (equivalent to `flex-grow: 2; flex-shrink: 1; flex-basis: 0%;`)
    - Three values: `flex: 2 1 200px;` (equivalent to `flex-grow: 2; flex-shrink: 1; flex-basis: 200px;`)

## Media Queries

> Media queries allow us to apply different styles based on the characteristics of the **device or viewport**. This is essential for creating responsive designs that adapt to various screen sizes.

The most used options are `min-width` and `max-width`, which allow us to target specific screen widths. And you can also combine them with `and` to create more complex queries.

<div class="grid" markdown>

```css
/* Styles for screens LARGER than 768px */
@media (min-width: 768px) {
    .container {
        flex-direction: row;
    }
}
```

```css
/* Styles for screens SMALLER than 768px */
@media (max-width: 768px) {
    .container {
        flex-direction: column;
    }
}
```

</div>

Another common (but slightly less used) option is `orientation`, which allows us to target devices based on their orientation (portrait or landscape).

## Container Queries

While media queries are based on the viewport size, container queries respond to the size of their **parent container**.

<div class="grid" markdown>

For this to work, the parent container must have the `container-type` property! Set it to `inline-size` (most common) or `size` (both width and height). Then you can use `@container` to define styles based on the container's size. <br><br>
The container does not need a name in most cases. But if you want to target a "grandparent" container, you can give it a name using `container-name` and then use that name: `@container my-container (min-width: 400px) { ... }`.

```css
/* Parent container */
.parent {
    container-type: inline-size;
}

/* Default styles for child elements */
.child {
    background-color: lightblue;
    padding: 1rem;
}

/* Styles for child elements when the parent container is at least 400px wide */
@container (min-width: 400px) {
    .child {
        background-color: lightgreen;
        padding: 2rem;
    }
}
```

</div>
