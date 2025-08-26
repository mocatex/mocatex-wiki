# <center>uv package manager</center>

<img src="../../assets/images/python/uv-logo.jpg" alt="uv-logo" width="50%" style="display: block; margin: auto">

> uv is a modern and fast package manager for Python projects, designed to simplify dependency management and project setup.

## 1. General Overview

### 1.1 Structure of a uv project

A uv project typically consists of the following uv-specific files and directories:

- `pyproject.toml`: contains the project metadata, dependencies, and build system requirements.
- `.python-version`: specifies the Python version to be used for the project.
- `uv.lock`: locks the exact versions of the dependencies to ensure reproducible builds. (get auto-generated when you install dependencies)
- `.venv/`: a directory that contains the virtual environment for the project. (get auto-generated when you install dependencies)

### 1.2 Installation and Setup

You can use brew to install uv:

```bash
brew install uv
```

To create a new uv project, navigate to your desired directory and run:

```bash
uv init my-project
```

!!! tip
    If you already have a project setup, you can run `uv init` (without a project name) in the root directory of your project to set it up with uv.

### 1.3 Managing Dependencies

To add or remove a dependency to your project, use the following command:

```bash
uv add <package-name>
uv remove <package-name>
```

This will update the `pyproject.toml` and `uv.lock` files accordingly.

!!! tip
    - You can also add it manually to the `pyproject.toml` file.
    - If you want to lock/sync manually, you can run `uv lock` or `uv sync` to manually update the `uv.lock` file or sync the virtual environment with the `pyproject.toml` file.

### 1.4 Activating the Virtual Environment

You can either use the fast and convenient command:

```bash
uv run your_file.py
```

Here uv will automatically activate and then deactivate the virtual environment for you.

or you can activate the virtual environment manually:

```bash
source .venv/bin/activate
```

This will activate the virtual environment, and you can run your Python scripts within this environment.
To deactivate the virtual environment, simply run:

```bash
deactivate
```
