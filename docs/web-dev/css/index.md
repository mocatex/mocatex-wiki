# CSS - The Basics

![CSS Logo](../../assets/images/web-dev/css-logo.png){width=20% .center}

> CSS (Cascading Style Sheets) is a stylesheet language used to describe the presentation of a document written in HTML or XML. It controls the layout, colors, fonts, and overall visual appearance of web pages.

## How to inlcude CSS in HTML

There are three main ways to include CSS in an HTML document:

1. :x: **Inline CSS**: Using the `style` attribute within HTML elements. -> Very bad

   ```html
   <p style="color: blue; font-size: 16px;">This is a blue paragraph.</p>
   ```

2. :warning: **Internal CSS**: Using a `<style>` tag within the `<head>` section of the HTML document. -> also bad

   ```html
   <head>
       <style>
           h1 {
               color: navy;
           }
       </style>
   </head>
   ```

3. :white_check_mark: **External CSS**: Linking to external CSS file(s) using the `<link>` tag within the `<head>` section. -> Best practice

    ```html
    <head>
         <link rel="stylesheet" href="styles.css">
    </head>
    ```

Then in those files you can write your CSS code!

## Basic CSS Syntax

CSS rules are made up of selectors and declarations. A selector targets HTML elements, and declarations define the styles to be applied.
Each declaration must end with a semicolon (`;`).

```css
selector {
    property: value;
    property: value;
}
```

You can find more indepth information about it in my [CSS Selectors Guide](./css-selectors.md).

## Cascading and Specificity

CSS follows the "cascading" principle, where multiple styles can apply to the same element.

**The most specific rule takes precedence.**
Specificity is determined by the type of selector used (inline styles > IDs > classes > elements).

Also, the order of the rules matters: later rules can override earlier ones if they have the same specificity.

## CSS Color Systems

### Named Colors

CSS supports a set of named colors that can be used directly in your styles. Examples include:

- `red`
- `green`
- `indigo`
- ...

most of the times, you will want to use more specific colors.

### RGB & RGBA Colors

**RGB** (Red, Green, Blue) colors are defined using the `rgb()` function, which takes three values ranging from 0 to 255.

```css
color: rgb(255, 0, 0); /* Red */
color: rgb(0, 255, 0); /* Green */
color: rgb(0, 0, 255); /* Blue */
color: rgb(206, 51, 234) /* Purple */
```

**RGBA** colors are similar to RGB but include an *alpha channel* for transparency, ranging from 0 (fully transparent) to 1 (fully opaque).

```css
color: rgba(255, 0, 0, 0.5); /* Semi-transparent Red */
```

### HEX & HEXA Colors

HEX colors are represented as a six-digit hexadecimal number, prefixed with a `#`. Each pair of digits represents the intensity of red, green, and blue, respectively.

```css
color: #FF0000; /* Red */
color: #00FF00; /* Green */
color: #0000FF; /* Blue */
color: #CE33EA; /* Purple */
```

**HEXA** colors are similar to HEX but include an additional two digits for the alpha channel (transparency).

```css
color: #FF000080; /* Semi-transparent Red */
```

!!! tip
    RGB(A) and HEX(A) are actually the same.
    In HEX(A) the values are just represented in hexadecimal format.
    -> 255 = FF, 0 = 00, 128 = 80, ...

#### Shorthand HEX

If each pair of digits in a HEX color are the same, you can shorten it to three digits.

```css
color: #FF3399; /* Full HEX */
color: #F39;    /* Shorthand HEX */
color: #F39A;   /* Shorthand HEX with alpha */
```

!!! warning "Important"
    The shorthand HEX(A) Format only if each pair of digits are the same!
    Even if a single pair is different, you must use the full six/eight-digit format.

## Common Text Properties

### text-align

The `text-align` property specifies the horizontal alignment of text within an element.

```css
p {
    text-align: left;    /* Aligns text to the left */
    text-align: center;  /* Centers the text */
    text-align: right;   /* Aligns text to the right */
    text-align: justify; /* Justifies the text -> adds space between words to align both edges */
}
```

### font-weight

The `font-weight` property specifies the weight (or boldness) of the font.

```css
p {
    font-weight: normal; /* Normal weight */
    font-weight: bold;   /* Bold weight */
    font-weight: 100;    /* Thin */
    font-weight: 900;    /* Extra Bold */
}
```

Numeral values range from 100 (thin) to 900 (extra bold), with `normal` equivalent to 400 and `bold` equivalent to 700.

!!! warning
    Not all fonts support all weight values!
    Commonly used values are `normal`, `bold`, `bolder`, and `lighter`.

### text-decoration

The `text-decoration` property specifies the decoration added to text.

```css
a {
    text-decoration: none;        /* No decoration */
    text-decoration: underline;   /* Underlines the text */
    text-decoration: #F75A1E wavy overline; /* Adds a wavy overline with a specific color */
    text-decoration: line-through;/* Adds a line through the text */
}
```

So you can combine colors, styles and lines.

The options for styles are `solid`, `double`, `dotted`, `dashed`, and `wavy`.

### line-height

The `line-height` property specifies the height of a line of text. -> Space between lines
It can be set using a number, length, or percentage.

```css
p {
    line-height: normal; /* Default line height */
    line-height: 1.5;    /* 1.5 times the font size */
    line-height: 20px;   /* Fixed height of 20 pixels */
    line-height: 150%;   /* 150% of the font size */
}
```

### letter-spacing

The `letter-spacing` property specifies the space between characters in text.

```css
p {
    letter-spacing: normal; /* Default spacing */
    letter-spacing: 2px;    /* Increases spacing by 2 pixels */
    letter-spacing: -1px;   /* Decreases spacing by 1 pixel */
}
```

### word-spacing

The `word-spacing` property is the same as `letter-spacing`, but it applies to the space between words.

### font-size

The `font-size` property specifies the size of the font.
It can be set using various units, such as pixels (`px`), ems (`em`), rems (`rem`), percentages (`%`), and more.

```css
p {
    font-size: 16px;  /* Fixed size of 16 pixels */
    font-size: 1.5em; /* 1.5 times the size of the parent element's font size */
    font-size: 150%;  /* 150% of the parent element's font size */
}
```

### font-family

The `font-family` property specifies the font(s) to be used for text.

```css
p {
    font-family: Arial, sans-serif; /* Uses Arial, falls back to any sans-serif font */
}
```

## CSS Units

CSS supports various units for specifying sizes, lengths, and other measurements.
There are two main categories: absolute and relative units.

### Absolute Units

Absolute units are fixed and do not change based on other elements or the viewport size.

- `px` (pixels): A pixel is a single dot on the screen. It is the most commonly used absolute unit.
- `cm` (centimeters): A centimeter is a metric unit of length. (same for `mm`, `in`, `pt`, `pc`) <br>
-> rarely used

### Relative Units

Relative units are based on other measurements, such as the size of the parent element or the viewport.

- `em`: Relative to the font size of the **parent element**. <br>
-> `2em` is twice the size of the parent element's font size.

    >If the parent has no specified font size, it inherits from its parent, and so on, until it reaches the root element (usually the `<html>` element).

- `rem`: Relative to the font size of the **root element** (usually the `<html>` element). <br>
-> `1.5rem` is 1.5 times the root element's font size.
This is often more predictable than `em`, especially for nested elements.
!!! info "Important"
    Use `rem` over `em` for more predictable sizing, especially for nested elements.

- `%`: Relative to the parent element's size. <br>
-> `50%` width means half the width of the parent element.
- `vw` (viewport width): Relative to 1% of the width of the viewport (browser window). <br>
-> `10vw` is 10% of the viewport width.
- `vh` (viewport height): Relative to 1% of the height of the viewport. <br>
-> `10vh` is 10% of the viewport height.
- `vmin`: Relative to 1% of the smaller dimension of the viewport (width or height).
- `vmax`: Relative to 1% of the larger dimension of the viewport (width or height).

!!! tip
    - Use `px` for precise control over element sizes.
    - Use `em` and `rem` for scalable typography and spacing.
    - Use `%`, `vw`, and `vh` for responsive layouts that adapt to different screen sizes.
