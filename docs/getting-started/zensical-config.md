---
tags:
    - zensical
    - configuration
    - setup

icon: lucide/settings
---

# Zensical Configuration

> Zensical uses a `zensical.toml` file to configure the backend. This file should be placed in the **root** of your project.

## Quick Setup and Start

Since we use the [uv package-manager](../python/uv-package-manager.md) for our dependencies, you can setup a new setup like this:

```bash
uv init my-first-zensical-project
uv add zensical
```

After that we need to initialize the zensical project:

```bash
uv run zensical new .
```

The only minimal required configuration in the `zensical.toml` file is the `site_name`:

```toml
[project]
site_name = "My First Zensical Project"
```

After that you can start the zensical server of your project with:

```bash
uv run zensical serve
```

Per default the server will be running on `http://localhost:8000` and you can open it in your browser to see your new zensical project. (You can change the port in the `zensical.toml` file, more on that later.)

When you want to test your project in a production-like environment, you can build the static files with:

```bash
uv run zensical build
```

You then can publish your site with GitHub Pages or any other static hosting provider. 

## Configuration Options

My recommendation is to just follow the [documentation](https://zensical.org/docs/setup/basics/) on the Zensical website since most of the default and recommended configuration options are pretty solid!

Take my base [default configuration](#my-recommended-base-configuration) as a starting point and then tweak it to your needs.

Yet, here I added the configurations options I actually consider tweaking for most projects:

### Basics

- `site_name`: The name of your site, shown in the header and title.
- `site_url`: The URL where your site will be hosted, used for generating absolute links.
- `site_description`: A short description of your site, used in meta tags for SEO.
- `site_author`: The name of the author of the site, used in meta tags and displayed in the footer.
- `copyright`: The copyright notice for your site, displayed in the footer.
- `dev_addr`: The address and port for the development server, e.g. `localhost:1234`.

### Colors

You can set the *primary* and *accent* colors of your site. Either in the simple way with predefined color names or with a custom css-file. I would recommend to just use the predefined color names:

<table>
    <tr>
        <td>
            <code>red</code>
        </td>
        <td>
            <img src="../assets/images/config-color-red.png" alt="Red color scheme" width="150">
        </td>
        <td>
            <img src="../assets/images/config-color-pink.png" alt="Pink color scheme" width="150">
        </td>
        <td>
            <code>pink</code>
        </td>
    </tr>
        <tr>
        <td>
            <code>purple</code>
        </td>
        <td>
            <img src="../assets/images/config-color-purple.png" alt="Purple color scheme" width="150">
        </td>
        <td>
            <img src="../assets/images/config-color-deep-purple.png" alt="Deep Purple color scheme" width="150">
        </td>
        <td>
            <code>deep purple</code>
        </td>
    </tr>
        <tr>
        <td>
            <code>indigo</code>
        </td>
        <td>
            <img src="../assets/images/config-color-indigo.png" alt="Indigo color scheme" width="150">
        </td>
        <td>
            <img src="../assets/images/config-color-blue.png" alt="Blue color scheme" width="150">
        </td>
        <td>
            <code>blue</code>
        </td>
    </tr>
        <tr>
        <td>
            <code>light blue</code>
        </td>
        <td>
            <img src="../assets/images/config-color-light-blue.png" alt="Light Blue color scheme" width="150">
        </td>
        <td>
            <img src="../assets/images/config-color-cyan.png" alt="Cyan color scheme" width="150">
        </td>
        <td>
            <code>cyan</code>
        </td>
    </tr>
        <tr>
        <td>
            <code>teal</code>
        </td>
        <td>
            <img src="../assets/images/config-color-teal.png" alt="Teal color scheme" width="150">
        </td>
        <td>
            <img src="../assets/images/config-color-green.png" alt="Green color scheme" width="150">
        </td>
        <td>
            <code>green</code>
        </td>
    </tr>
        <tr>
        <td>
            <code>light green</code>
        </td>
        <td>
            <img src="../assets/images/config-color-light-green.png" alt="Light Green color scheme" width="150">
        </td>
        <td>
            <img src="../assets/images/config-color-lime.png" alt="Lime color scheme" width="150">
        </td>
        <td>
            <code>lime</code>
        </td>
    </tr>
        <tr>
        <td>
            <code>yellow</code>
        </td>
        <td>
            <img src="../assets/images/config-color-yellow.png" alt="Yellow color scheme" width="150">
        </td>
        <td>
            <img src="../assets/images/config-color-amber.png" alt="Amber color scheme" width="150">
        </td>
        <td>
            <code>amber</code>
        </td>
    </tr>
        <tr>
        <td>
            <code>orange</code>
        </td>
        <td>
            <img src="../assets/images/config-color-orange.png" alt="Orange color scheme" width="150">
        </td>
        <td>
            <img src="../assets/images/config-color-deep-orange.png" alt="Deep Orange color scheme" width="150">
        </td>
        <td>
            <code>deep orange</code>
        </td>
    </tr>
</table>

### Logo & Icons

> Those images can either be a user provided image located in the `docs` folder or any [icon bundled](https://zensical.org/docs/authoring/icons-emojis/#included-icon-sets) with the theme

- `logo`: The logo of your site, shown in the header.
- `favicon`: The favicon of your site, shown in the browser tab.

### Navigation

You can set a custom navigation for your site within the `zensical.toml` file. But I would recommend letting zensical generate the navigation for you based on your file structure. You can override things like the title of a page with a `title` field in the [frontmatter](./writing-guide.md#metadata-frontmatter) of your markdown files.

Now settings that are useful yet not needed for all projects:

```toml
[project.theme]
featiures = [
    "navigation.sections", # subdirs are displayed as sections instead of expandables
]
```

### Social Media Links

You can add them in the `zensical.toml` file and they will be displayed in the footer of your site:

```toml
[[project.extra.social]]
icon = "fontawesome/brands/instagram"
link = "link-to-your-social-profile"
name = "Name for accessibility"
```

## My Recommended Base Configuration

Here is a ready-to-use copy-paste base configuration for you to start with. You can just copy this into your `zensical.toml` file and then tweak it to your needs:

```toml
[project] # Metadata and general settings
site_name = "mocatex Wiki"
site_url = "https://mocatex.github.io/mocatex-wiki/"
repo_url = "https://github.com/mocatex/mocatex-wiki"
repo_name = "mocatex/mocatex-wiki"
site_description = "Personal wiki for stuff I learn"
site_author = "mocatex"
copyright = "&copy; 2026 mocatex"
docs_dir = "docs"
dev_addr = "localhost:2020"

extra_css = ["assets/styles/overrides.css"]

extra_javascript = [
    "https://unpkg.com/tablesort@5.3.0/dist/tablesort.min.js",
    "javascripts/tablesort.js",
]

[project.theme]
language = "en"

# features of Zensical
features = [
    "navigation.instant",          # nav without page reloads
    "navigation.instant.prefetch", # prefetch pages on hover
    "navigation.instant.progress", # show progress bar on page load
    "navigation.tracking",         # track user navigation in local storage
    "navigation.tabs",             # top level dir are displayed as tabs in the header
    "navigation.tabs.sticky",      # keep the tabs in the header sticky on scrollstead of expandables
    "navigation.expand",           # expand all dirs per default in the sidebar
    "navigation.path",             # add breadcrumbs to the top of the page
    "navigation.prune",            # only visible nav items are rendered in DOM
    "navigation.indexes",          # index.md files become the default page for a directory
    "toc.follow",                  # toc is always visible
    "navigation.top",              # add a "back to top" button on the bottom right of the page
    "search.highlight",            # highlight search terms in the page
    "navigation.footer",           # ad links to the next and previous page at the end of each page

    "content.code.copy",     # add a copy button to code blocks
    "content.code.select",   # add selection button to highlight hovered line
    "content.code.annotate", # clickable popups for code annotations

    "content.tabs.link", # linkable content tabs

    "content.footnote.tooltips", # show footnotes tooltips

]

# Palette toggle for LIGHT mode
[[project.theme.palette]]
media = "(prefers-color-scheme: light)"
scheme = "default"
primary = "deep purple"
accent = "deep purple"
toggle.icon = "material/brightness-7"
toggle.name = "Switch to dark mode"

# Palette toggle for DARK mode
[[project.theme.palette]]
media = "(prefers-color-scheme: dark)"
scheme = "slate"
primary = "deep purple"
accent = "deep purple"
toggle.icon = "material/brightness-3"
toggle.name = "Switch to light mode"

[project.theme.icon]
logo = "material/text-box-search-outline"

# Admonition blocks
[project.markdown_extensions.admonition]
[project.markdown_extensions.pymdownx.details]
[project.markdown_extensions.pymdownx.superfences]
custom_fences = [
    { name = "mermaid", class = "mermaid", format = "pymdownx.superfences.fence_code_format" },
]

# Code blocks with syntax highlighting
[project.markdown_extensions.pymdownx.highlight]
anchor_linenums = true
line_spans = "__span"
pygments_lang_class = true
[project.markdown_extensions.pymdownx.inlinehilite]
[project.markdown_extensions.pymdownx.snippets]

# Content Tabs
[project.markdown_extensions.pymdownx.tabbed]
alternate_style = true
[project.markdown_extensions.pymdownx.tabbed.slugify]
object = "pymdownx.slugs.slugify"
kwds = { case = "lower" }

# Tables
[project.markdown_extensions.tables]

# Footnotes
[project.markdown_extensions.footnotes]

# Text Formatting
[project.markdown_extensions.pymdownx.caret]
[project.markdown_extensions.pymdownx.keys]
[project.markdown_extensions.pymdownx.mark]
[project.markdown_extensions.pymdownx.tilde]

# Grid & Grid-Cards
[project.markdown_extensions.attr_list]
[project.markdown_extensions.md_in_html]

# Emojis & Icons
[project.markdown_extensions.pymdownx.emoji]

# Images
[project.markdown_extensions.pymdownx.blocks.caption]

# Lists
[project.markdown_extensions.def_list]
[project.markdown_extensions.pymdownx.tasklist]
custom_checkbox = true
```