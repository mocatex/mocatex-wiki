---
tags:
    - getting started
    - writing guide
    - markdown
    - formating

icon: material/pencil-plus
---

# Writing Guide

### TODO List:

- Instant Preview


## Metadata (Frontmatter)

Every markdown file can have a frontmatter section at the top, which is used to provide metadata about the document. This metadata is used by Zensical to add additional features to the page.

You can add the frontmatter section by adding a YAML block at the top of your markdown file:

```yaml
---
# Your metadata goes here
---
```

### Title

WIth this you overwrite the `H1` title that would be used for the navigation.

```yaml
title: "Your Title"
```

### Tags

These tags are shown at thebottom of each page and can be used when filtering searches. Also in the future there will be a dedicated tag page.

```yaml
tags:
  - tag1
  - tag2
```

### Page Icon

The icons has to be from the [included icon library](https://zensical.org/docs/authoring/icons-emojis/#included-icon-sets):

```yaml
icon: lucide/smile
```

You have the following libraries built in:

- `lucide/`: [Lucide Icons](https://lucide.dev/icons/) -> for general icons
- `simple/`: [Simple Icons](https://simpleicons.org/) -> for brands and products
- `octicons/`: [Octicons](https://primer.style/octicons/) -> for GitHub related icons
- `material/`: [Material Symbols](https://pictogrammers.com/library/mdi/) -> for general icons

## Admonition Blocks

Admonition blocks are used to highlight important information, warnings, tips, etc. They are rendered as colored boxes with an icon and a title.

You can create an admonition block by using the following syntax:

```markdown
!!! type "Title"
    Your content goes here.
```

- using an empty string `""` for the title will hide the title and only show the content in the box.
- using `???` will create a collapsible block that is collapsed by default. Add a `+` at the end to make it expanded by default: `???+ type "Title"`
- add a `inline` or `inline end` between *type* and *title* to make the block inline. *inline end* will align the block to the *right*.
-> **Important**: The admonition block has to be declared **before** the content that should be inline with it.

Here are the available types:

!!! note "note"

!!! abstract "abstract"

!!! info "info"

!!! tip "tip"

!!! success "success"

!!! question "question"

!!! warning "warning"

!!! failure "failure"

!!! danger "danger"

!!! bug "bug"

!!! example "example"

!!! quote "quote"

## Code Blocks

You can add code blocks by using triple backticks ``` before and after the code block and specifying the language for syntax highlighting:

````markdown
```python
print("Hello, World!")
```
````

- You can add a **title** to the code block by adding `title="your title"` after the language.
- You can enable **line numbers** by adding `linenums="1"` after the language. ("1" means start from 1)
- You can highlight specific lines by adding `hl_lines="1 2 3"` after the language to highlight lines 1, 2 and 3.
- **Inline code blocks** can be highlighted by adding a shebang `!#` followed by the language code: #!python print("Hello, World!")
- Code Annotations can be added by adding a number within this structure `(number)!` within a block comment of the corresponding language. Then you cann them below the code (**outside** of it) block in a numbered list with the same number followed by the annotation text. For example:

````text
```python
def add(a, b):
    return a + b  # (1)!
```

1. This function adds two numbers together.
````

Here is a full example of a code block with all these features:

````markdown
```python title="Example Code" linenums="1" hl_lines="2 3"
def add(a, b):
    return a + b  # (1)!
```

1. This function adds two numbers together.
````

And here is how it looks like:

```python title="Example Code" linenums="1" hl_lines="2 3"
def add(a, b):
    return a + b  # (1)!
```

1. This function adds two numbers together.

## Content Tabs

You can create content tabs by using the following syntax:

```markdown
=== "Tab 1"
    Content for Tab 1 goes here. Also supports markdown formatting.

=== "Tab 2"
    Content for Tab 2 goes here. Same here
```

If you add later again tabs with the same name, they will be in sync, meaning if you switch the tab in one of them, it will switch in all of them.

Here is an example of content tabs:

=== "Python"

    ```python
    print("Hello, World!")
    ```

=== "JavaScript"

    ```javascript
    console.log("Hello, World!");
    ```

And here again tabs with the same name that are in sync:

=== "Python"

    ```python
    def add(a, b):
        return a + b
    ```

=== "JavaScript"

    ```javascript
    function add(a, b) {
        return a + b;
    }
    ```

## Mermaid Diagrams

You can add [Mermaid diagrams](https://mermaid.js.org/intro/) by using the following syntax:

````markdown
```mermaid
    here goes your mermaid code
```
````

## Footnotes

You can add footnotes by using the following syntax:

```markdown
This is some text with a footnote.[^1]
[^1]: This is the footnote text.
```

Here is an example of a footnote.[^1]

[^1]: This is the footnote text.

## Formatting Text

Zensical supports six different additional text formatting options that are not supported by default in markdown:

- **Highlighting**: With `==` you can ==highlight text==: `==highlighted text==`
- **Underline**: With `^^` you can ^^underline text^^: `^^underlined text^^`
- **Strikethough**: With `~~` you can ~~strikethough text~~: `~~strikethough text~~`
- **Superscript**: With `^` you can add superscript ^text^: `superscript^text^`
- **Subscript**: With `~` you can add subscript ~text~: `subscript~text~`
- **Keyboard**: With `++` you can add keyboard-like commands ++alt+f4++: `++alt+f4++`
-> you can find all the supported formatting options in the [pymdownx extensions](https://facelessuser.github.io/pymdown-extensions/extensions/keys/#extendingmodifying-key-map-index) documentation.

## Grid System & Grid Cards

### Grid System

You cann add **block elements** (Admonitions, code blocks, content tabs, etc.) in a griod layout like this: -> **NO Indentation**

````html
<div class="grid" markdown>
```python title="Grid Example Code"
def product(a, b):
    return a * b
```

!!! info "Grid Example Admonition"
    This is an example of an admonition block within a grid.

=== "Grid Example Tab 1"
    This is the content of the first tab within the ==grid==.
    
=== "Grid Example Tab 2"
    This is the content of the second tab within the *grid*.

| duck around | find out |
| ----------- | -------- |
| less        | less     |
| more        | more     |
</div>
````

this will look like this:

<div class="grid" markdown>
```python title="Grid Example Code"
def product(a, b):
    return a * b
```

!!! info "Grid Example Admonition"
    This is an example of an admonition block within a grid.

=== "Grid Example Tab 1"
    This is the content of the first tab within the ==grid==.
    
=== "Grid Example Tab 2"
    This is the content of the second tab within the *grid*.

| duck around | find out |
| ----------- | -------- |
| less        | less     |
| more        | more     |

</div>

### Grid Cards

Grid cards are a special type of grid element since they have a card-like design. They can can be made by some text or block elements.

````html
<div class="grid cards" markdown>
- This is the first item in the grid card.
- They also support :smile: emojis and other markdown formatting.
- For example, you can add **bold** text and *italic* text.
- But also zensical specific like ==highlighting== and ^^underlining^^.
- Here is some code within a grid card:

    ```python
    def greet(name):
        return f"Hello, {name}!"
    ```

- or an admonition block

    !!! tip "Grid Card Tip"
        This is an example of an admonition block within a card.
        
</div>
````

This will look like this:

<div class="grid cards" markdown>
- This is the first item in the grid card.
- They also support :smile: emojis and other markdown formatting.
- For example, you can add **bold** text and *italic* text.
- But also zensical specific like ==highlighting== and ^^underlining^^.
- Here is some code within a grid card:

    ```python
    def greet(name):
        return f"Hello, {name}!"
    ```

- or an admonition block

    !!! tip "Grid Card Tip"
        This is an example of an admonition block within a card.

</div>

## Images

You can add **css attributes** to images by adding a `{}` after them and adding the attributes **with a space after the opening bracket**:

- `{ align=left/right }`: aligns the image left or right to a text -> Image needs to be declared **before** the text to be aligned with it.
- `{ width=100px }`: sets the width of the image to 100px
- `{ .center }`: centers the image (custom css class)
- `{ loading=lazy }`: adds lazy loading to the image -> only needed if you have a lot and big images on a page to improve performance
- **caption**: You can add one by putting this directly after the image:

    ```markdown
    /// caption
    [Image caption here]
    ///
    ```

## Lists

Markdown already supports ordered and unordered lists by default.
Zensical adds a nice support for **nested lists** with different bullet points and numbering styles. (no special syntax needed, just indent the list items)

Also it supports **definition lists** with the following syntax:

<div class="grid" markdown>

```markdown
Term 1
: Definition for term 1

Term 2
: Definition for term 2
```

Term 1
: Definition for term 1

Term 2
: Definition for term 2
</div>

And **task lists** with checkboxes that can be checked off by clicking on them:


<div class="grid" markdown>

```markdown
- [ ] Task 1
    - [ ] Subtask 1
    - [x] Subtask 2
- [x] Task 2
```

- [ ] Task 1
    - [ ] Subtask 1
    - [x] Subtask 2
- [x] Task 2

</div>