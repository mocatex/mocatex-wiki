# HTML - All You Need to Know

## 1. Introduction to HTML

- **Definition**: HTML (**H**yper**T**ext **M**arkup **L**anguage) is the standard language for structuring content on the web.
- **How it works**:

    - HTML describes the structure of a web page using *elements* enclosed in tags (`<tag>`).
    - Elements can contain text, attributes, and other elements.
    - Browser parses HTML into a **DOM (Document Object Model)** tree.

### Minimal Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>My Page</title>
</head>
<body>
  <h1>Hello World</h1>
  <p>This is my first HTML page.</p>
</body>
</html>
```

!!! tip
    In VSCode, use the `!` + `Tab` shortcut to generate a basic HTML boilerplate quickly.

---

## 2. Basic Structure & Elements

### Headings and Text

- `<h1>`–`<h6>` -> Headings (hierarchical, `<h1>` most important).
- `<p>` -> Paragraph.
- `<span>` -> Inline container for styling/text snippets.
- `<br>` -> Line break.

### Lists

- `<ul>` -> Unordered list (bullets).
- `<ol>` -> Ordered list (numbers).
- `<li>` -> List item.
- `<dl>`, `<dt>`, `<dd>` -> Description list.

### Links & Media

- `<a href="...">` -> Hyperlink.
    You can add custom a custom link-text between the tags.
- `<img src="..." alt="...">` -> Image.
- `<video>`, `<audio>` -> Embedded media.

!!! Note
    If you want to open links in a new tab, add `target="_blank"` to the `<a>` tag.

---

## 3. Forms

Forms allow user input and interaction.

### Basic Structure

```html
<form action="/submit" method="post">
  <label for="username">Username:</label>
  <input id="username" name="username" type="text" required>
  
  <label for="password">Password:</label>
  <input id="password" name="password" type="password" required>
  
  <button type="submit">Login</button>
</form>
```

### Key Elements - Forms

- `<input>` – versatile element with `type` attribute:
    - `text`, `password`, `email`, `number`, `date`, `checkbox`, `radio`, `file`, `submit`.
- `<textarea>` – multiline input.
- `<select>` with `<option>` – dropdowns.
- `<label>` – improves accessibility by binding text to input.
- `<fieldset>` & `<legend>` – group form controls.
- Validation attributes: `required`, `min`, `max`, `pattern`.

!!! tip
    You can find more about forms in [my forms guide](./html-forms.md).

---

## 4. Tables

Tables organize data in rows/columns.

### Example

```html
<table>
  <caption>Student Grades</caption>
  <thead>
    <tr>
      <th>Name</th>
      <th>Grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Alice</td>
      <td>A</td>
    </tr>
    <tr>
      <td>Bob</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
```

### Key Elements - Tables

- `<table>` – container.
- `<tr>` – table row.
- `<td>` – table data cell.
- `<th>` – header cell.
- `<thead>`, `<tbody>`, `<tfoot>` – structure sections.
- `<caption>` – table title.

Use tables **only for data** (not for layout).

---

## 5. Semantic & Structural Elements (When to Use What)

Modern HTML introduces **semantic elements** that give meaning to structure.

- `<header>` -> Page or section header.
- `<footer>` -> Page or section footer.
- `<nav>` -> Navigation links.
- `<main>` -> Main unique content of page.
- `<article>` -> Self-contained content (e.g., blog post).
- `<section>` -> Thematic grouping of content.
- `<aside>` -> Side content (e.g., sidebar, ads, tips).
- `<div>` -> Generic block-level container (no semantic meaning). Use only if no better element fits.
- `<span>` -> Generic inline container.

### Best Practices

- Prefer **semantic elements** (`<nav>`, `<section>`) -> improves accessibility & SEO.
- Use `<div>` only as fallback for grouping without specific meaning.
- Don’t over-nest divs (“div soup”).

---

## 6. Other Useful HTML Elements

- `<meta>` – metadata (charset, viewport, SEO info).
- `<link>` – external resources (CSS).
- `<script>` – JavaScript inclusion.
- `<iframe>` – embed external content.

---

## 7. Accessibility & Good Practices

- Always provide **alt text** for images.
- Use **labels** for form inputs.
- Follow **heading hierarchy** (`<h1>` then `<h2>`, …).
- Keep HTML **clean, semantic, and minimal**.