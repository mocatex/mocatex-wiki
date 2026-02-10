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

Take my base default configuration as a starting point and then tweak it to your needs:

```toml
add code later here!!!
```

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

You can set a custom navigation for your site within the `zensical.toml` file. But I would recommend letting zensical generate the navigation for you based on your file structure. You can override things like the title of a page with a `title` field in the [frontmatter]() of your markdown files.

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

