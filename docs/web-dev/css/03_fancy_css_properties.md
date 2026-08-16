---
title: "Fancy CSS"
icon: lucide/wand-sparkles
---

# Fancy CSS Properties

This is just a collection of some fancy CSS properties that can be used to create interesting effects and animations.

## Opacity

Pretty simple: the `opacity` property allows us to set the transparency of an **entire** element. It can take a value between `0` (completely transparent) and `1` (completely opaque).

## Position

The `position` property allows us to control the positioning of an element in relation to its parent or the viewport. It can take several values:

- `static`: The default value. The element is positioned according to the normal flow of the document.
- `relative`: The element is positioned **relative to its normal position**. We can use the `top`, `right`, `bottom`, and `left` properties to adjust its position. The element will still occupy its original space in the document flow.
- `absolute`: The element is positioned relative to its nearest positioned ancestor (not static). If there is no such ancestor, it will be positioned relative to the initial containing block (usually the viewport).
- `fixed`: The element is positioned relative to the viewport, which means it stays in the same position even when the page is scrolled.
- `sticky`: The element is positioned based on the user's scroll position. It toggles between `relative` and `fixed`, depending on the scroll position. It is treated as `relative` until it crosses a specified threshold, at which point it becomes `fixed`. This is useful for creating sticky headers or elements that remain visible while scrolling.

## Z-Index

The `z-index` property controls the stacking order of elements that overlap. Elements with a higher `z-index` value will appear in front of elements with a lower value. It only works on positioned elements (those with a `position` value other than `static`).

## Box-Shadow

<div class="grid" markdown>

The `box-shadow` property allows us to add shadow effects to an element. Can also be used to create a "glow" effect by using a spread radius and a color with some transparency.

```css
div {
    /* offset-x offset-y blur-radius spread-radius color */
    box-shadow: 10px 10px 5px 0px red;
}
```

</div>

## Transitions

<div class="grid" markdown>

The `transition` property allows us to create smooth animations when a property changes from one value to another. It can be used to animate changes in properties like `color`, `background-color`, `width`, `height`, and more. <br>
For `timing-function`, we can use values like `ease`, `linear`, `ease-in`, `ease-out`, and `cubic-bezier(...)` to control the speed curve of the transition.

```css
/* Example of a transition */
.button {
    background-color: blue;
    /* property duration timing-function delay */
    transition: background-color 0.3s ease 2s;
}
.button:hover {
    background-color: red;
}
```

</div>

## Transforms

The `transform` property allows us to apply various transformations to an element, such as scaling, rotating, translating, and skewing. It can take several functions as values:

- `scale(x, y)`: Scales the element by the specified factors along the X and Y axes.
- `rotate(angle)`: Rotates the element by the specified angle (in degrees or radians).
- `translate(x, y)`: Moves the element by the specified distances along the X and Y axes.
- `skew(x-angle, y-angle)`: Skews the element by the specified angles along the X and Y axes.
- `matrix(a, b, c, d, e, f)`: Applies a 2D transformation using a transformation matrix.

<div class="grid" markdown>

Since this is a "standard" property we can *animate* it using the `transition` property. For example, we can create a simple hover effect that scales and translates an element when the user hovers over it.

```css
.box {
    width: 100px; height: 100px;
    transition: transform 0.3s ease;
}

.box:hover {
    transform: scale(1.2) translate(20px, 20px);
}
```

</div>

## Background

We can use the `background` property to set the background color, image, position, size, and other properties of an element's background.

```css
div {
    background-color: lightblue; /* Set background color */
    background-image: url('image.jpg'); /* Set background image */
    background-position: center; /* Set background position */
    background-size: cover; /* Set background size */
    /* cover: cover the entire element, contain: fit the image inside the element */
}
```

## Backdrop-Filter

<div class="grid" markdown>

The `backdrop-filter` property allows us to apply graphical effects to the area behind an element. It can be used to create blur effects, adjust brightness, contrast, and more. For that the element must have some transparency (e.g. using `opacity` or `rgba` colors).

```css
div {
    backdrop-filter: blur(5px);
    backdrop-filter: brightness(0.5);
    backdrop-filter: contrast(2);
}
```

</div>

## Font-Family

<div class="grid" markdown>

The `font-family` property specifies the font to use for an element's text. It can take a list of font names, and the browser will use the first available font.

```css
div {
    font-family: 'Arial', 'Helvetica', sans-serif;
}
```

</div>

!!! note ""
    Fonts can be very expensive. A good source for free fonts is [Google Fonts](https://fonts.google.com/).
