---
title: "Layout Architecture"
icon: material/layers-search
---

# CSS Layout Architecture - A Practical Recipe

> Here is a practial rundown of how to structure your CSS (mainly [Grid-System](03_fancy_css_properties.md#grid-system) and [Flexbox](04_css-flexbox.md)) to create a responsive layout architecture.

## 1. The HTML Foundation

Write and finish the HTML **first** before any CSS! It is tempting to start styling before so you have something to see, but it is a bad habit. The site shoudl be readable and functional without any CSS:

- Correct landmarks: `header`, `nav`, `main`, `footer`, `aside`, `section`, `article`
- Correct heading structure: `h1` -> `h2` -> `h3` (no skipping)

It's a pain to stick through, but worth it in the long run.

## 2. Mobile-First

<div class="grid" markdown>

> It's not a trend! It's the right way to build responsive layouts. Start with the smallest screen size and then add styles for larger screens. This way you can save unnecessary overrides!

```css
/* additive -> adding capability as space allows */
.layout { display: flex; flex-direction: column; }

@media (min-width: 48rem) {
    .layout { flex-direction: row; }
}
```

</div>

!!! warning "Don't overuse media queries"
    They can get messy quickly! For intrinsic responsiveness (adapting to the content), use `flex-wrap`, `minmax()`, `clamp()`, etc.

## 3. CSS Layer architecture

<div class="grid" markdown>

CSS layers allow us to define the order of our stylesheets and avoid specificity wars. We can use `@layer` to define layers and `@import` to import stylesheets into specific layers. <br><br>
The **last** layer has the highest specificity, while selectors outside of the layers still win.

```css
/* Define layers */
@layer reset, tokens, base, layout, components, utilities;

/* Import stylesheets into layers */
@import "reset.css" layer(reset);
@import "tokens.css" layer(tokens);

@layer base {
    body {
        font-family: sans-serif;
        line-height: 1.5;
    }
}
```

</div>

- `reset`: usually just import the reset.css file
- `tokens`: define your design tokens/parameters (colors, spacing, typography, etc.)
- `base`: define your base styles (typography, body, headings, etc.)
- `layout`: define your layout styles (grid, flexbox, etc.)
- `components`: define your component styles (buttons, cards, modals, etc.)
- `utilities`: define your utility classes (spacing, text alignment, etc.)

## 4. macro/micro split -> grid/flexbox

1. **Page sekeleton**: Use **^^CSS Grid^^** to define the overall layout of the page. (header, sidebar, main, footer, etc.)
2. **Component layout**: Inside the grid areas, use **^^Flexbox^^** to define the layout of individual components. (buttons, cards, modals, etc.)

## 5. Flexbox recipe

**^^Step 1: Identify the axis^^** - Determine the main axis (row or column) and the cross axis (column or row) for your flex container.

**^^Step 2: Container Properties^^**

```css
.container {
    display: flex;
    flex-direction: row;             /* the axis, decided in step 1 */
    justify-content: space-between;  /* distribution along the MAIN axis */
    align-items: center;             /* alignment along the CROSS axis */
    flex-wrap: wrap;                 /* allow reflow instead of overflow/squeeze */
    gap: var(--space-3);             /* spacing — see step 3 */
}
```

**^^Step 3: Spacing^^** - Use the `gap` property to define spacing between flex items. Never use hacky margins for spacing! Use design tokens or CSS variables to maintain consistency.

**^^Step 4: Fix Overflow^^** - Use `min-width: 0` so that flex items can shrink properly and avoid overflow issues. This is especially important for text content that may not fit within its container. (URLs, filename, etc.)

**^^Step 5: Container Queries^^** - especially for components, use container queries to adapt the layout based on the size of the container rather than the viewport.
