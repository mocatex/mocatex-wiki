

# MkDocs Material

![mkdocs-material-logo](../../assets/images/web-dev/mkdocs-material-logo.png){width="20%" .center}

!!! info
    MkDocs Material is a popular static site generator that transforms Markdown files into static website. It is built on top of MkDocs, a static site generator designed for project documentation.

## 1. How to setup a MkDocs Material project

### Step 0: Install Python and uv package manager

Make sure you have Python installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

We are going to use the uv package manager for our project.
More about it and how to set it up can be found in my [uv documentation](../../python/uv-package-manager.md).

### Step 1: Setup Environment

- Create a `mkdocs.yml` file in the root of your project. This file will contain the configuration for your MkDocs site.
- add the following packages to your project via `uv add <package-name>` or manually to your `pyproject.toml` file:
    - `mkdocs-material`
    - `mkdocs-awesome-pages-plugin`
    - `mkdocs-section-index`
    - `pymdown-extensions`

### Step 2: Configure mkdocs.yml

Here is a basic configuration for `mkdocs.yml`:

```yaml
site_name: My Documentation Site
site_url: https://example.com
site_description: A place where I write down stuff I learn
site_author: Your Name

theme:
  name: material
  features:
    - feature one
    - feature two
    - ...
    palette:
        - scheme: default # light mode
            primary: indigo
            accent: indigo
            toggle:
                icon: material/brightness-7
                name: Switch to dark mode
        - scheme: slate # dark mode
            primary: indigo
            accent: indigo
            toggle:
                icon: material/brightness-3
                name: Switch to light mode
    
    language: en

plugins:
    - plugin one
    - plugin two
    - ...

markdown_extensions:
    - extension one
    - extension two
    - ...

extra:
    social: []

extra_css:
    - css/custom.css

extra_javascript:
    - js/custom.js
```

More info about the different colors and icons can be found [here](https://squidfunk.github.io/mkdocs-material/setup/changing-the-colors/).

#### Step 2.1: My favorite features

| Feature                           | Description                                                              |
| --------------------------------- | ------------------------------------------------------------------------ |
| **`navigation.instant`**          | Enables instant navigation between pages without a full page reload.     |
| **`navigation.instant.prefetch`** | Prefetches linked pages on hover to speed up navigation.                 |
| **`navigation.instant.progress`** | Displays a progress bar at the top of the page during navigation.        |
| **`navigation.path`**             | Adds Breadcrumb Navigation to the top of each page.                      |
| **`navigation.tabs`**             | Adds tabbed navigation for top-level pages.                              |
| **`navigation.tabs.sticky`**      | Makes the tab navigation sticky at the top of the viewport.              |
| **`navigation.path`**             | Adds breadcrumb navigation to the top of each page.                      |
| **`navigation.top`**              | Adds a `Back to top` button when scrolling up.                           |
| **`navitgation.sections`**        | Groups navigation items but keep nested on mobile                        |
| **`navigation.expand`**           | Expand all navigation sections by default                                |
| **`search.suggest`**              | Provides search suggestions as you type in the search box.               |
| **`search.highlight`**            | Highlights search terms on the page when navigating from search results. |
| **`search.share`**                | Adds a shareable link icon next to search results.                       |
| **`content.code.copy`**           | Adds a copy button to code blocks for easy copying of code snippets.     |

#### Step 2.2: My favorite plugins

| Plugin              | Description                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------ |
| **`search`**        | Adds a search box to your site, allowing users to search for content.                                  |
| **`awesome-pages`** | Enable custom or auto setup of navigation                                                              |
| **`section-index`** | Allows you to use `index.md` files as section landing pages.                                           |
| **`tags`**          | Adds tag support to your documentation. [How to use them?](../../getting-started/how-to-write.md#tags) |

#### Step 2.3: My favorite markdown extensions

| Extension                     | Description                                                                                                                                                                                         |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`admonition`**              | Adds support for admonition blocks (info, warning, tip, etc.). [More info](../../getting-started/how-to-write.md#admonition-blocks)                                                                 |
| **`toc`**                     | Automatically generates a table of contents for your pages. (add `permalink: true` to add a link icon next to each heading)                                                                         |
| **`pymdownx.blocks.details`** | Adds support for collapsible details blocks. [More info](../../getting-started/how-to-write.md#detail-blocks)                                                                                       |
| **`pymdownx.superfences`**    | Enhances code blocks with additional features like titles, line numbers, and highlighted lines. [More info](../../getting-started/how-to-write.md#code-blocks)                                      |
| **`pymdownx.magiclinks`**     | Automatically converts URLs / Emails and phone numbers into clickable links.                                                                                                                        |
| **`attr_list`**               | Allows you to add attributes to Markdown elements using: `{#id .class key=val}`                                                                                                                     |
| **`pymdownx.emoji`**          | Adds support for emojis using `:emoji_name:` syntax. -> [how to configure right emoji set](https://facelessuser.github.io/pymdown-extensions/extensions/emoji/#supported-emoji-providers-twemoji_1) |

### Step 3: Setup navigation via .pages file

If the navigation from `awesome-pages` is not enough you can setup a .pages file in every directory where you want to customize the navigation.

Here are all the needed options in a nutshell:

```yaml
nav:
  - index.md # show this file first
  - Vue App: vueapp.md # rename file in navigation
  - React App: reactapp.md
  - Subsection: # create custom subsection
    - nested-page.md # nested page
  - ... # this will add all other files in this directory automatically
  - _hidden-page.md # hide this page from navigation
```

### Step 4: create tags

There are two ways to create tags for your documentation:

1. at the top of every markdown file you can add:

    ```yaml
    ---
    tags:
    - tag1  
    - tag2
    - ...
    ---
    ```

2. or you can create a `.meta.yml` file in every directory where you want to add tags for all the markdown files in this directory and **all its subdirectories**:

    ```yaml
    tags:
    - tag1
    - tag2
    - ...
    ```

### Step 5: Build and serve your site locally

Since we are using the `uv` package manager, we can use it to run mkdocs commands.

#### Serve locally

!!! info
    Serving is like a fast preview of your site. It will automatically rebuild and refresh the site when you make changes to your markdown files.
    But it is not optimized for production.

```bash
uv run mkdocs serve -a 0.0.0.0:8000
```

> This will start a local server at `http://localhost:8000` where you can view your site.

#### Build for production

```bash
uv run mkdocs build --strict
```

> This will generate the static files for your site in the `site/` directory. So add this folder to your `.gitignore` file.

## 2. Deploy your site to GitHub Pages

### Step 1: Create a pages.yaml workflow

Create a new file at `.github/workflows/pages.yaml`.
This workflow will automatically build and deploy your site to GitHub Pages whenever you push changes to the `main` branch.

Here is a example workflow:
It uses GutHub Artifacts to store the built site and then deploys it using the `actions/deploy-pages` action.

An old method would be to use a gh-pages branch, but this is now not recommended anymore.

```yaml
name: Build & Deploy MkDocs
on:
  push:
    branches: [ main ]
  workflow_dispatch:


permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install mkdocs-material pymdown-extensions mkdocs-awesome-pages-plugin mkdocs-section-index
      - run: mkdocs build --strict
      - uses: actions/upload-pages-artifact@v4
        with:
          path: ./site
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: github-pages
    steps:
      - id: deploy
        uses: actions/deploy-pages@v4
```

!!! attention Important
    - Make sure to activate GitHub Pages in the settings of your repository and set the source to "GitHub Actions".
    - Set it to deploy from a github action workflow and *not* from a branch.

Now you can add, change or delete markdown/yaml files and push them to your `main` branch.

Happy documenting! :tada:
