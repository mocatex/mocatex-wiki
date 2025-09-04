# Pre-Commit Framework

![pre-commit-logo](../../assets/images/tooling/pre-commit-logo.jpg)

!!! tldr
    Pre-commit is a framework for managing and maintaining multi-language pre-commit [git-hooks](../../git/git-hooks.md) 
    It allows you to run checks on your code before committing it to your version control system.

## 1. Install pre-commit

**for macOS (via brew)**

```bash
brew install pre-commit
``` 

**general installation (via pipx)**

```bash
pipx install pre-commit
```

## 2. Create a configuration file

Create a `.pre-commit-config.yaml` file in the **root of your repository** with the following content:

```yaml
repos:
    - repo: 'link-to-repo'
        rev: 'version-or-branch' # used for reproducibility
        hooks:
            - id: 'hook-id' # is pre-defined in the repo
            args: ['--arg1', '--arg2'] # optional
            exclude: ^(pattern)$ # optional, regex pattern to exclude files, direct file names also possible -> `exclude: file.txt`
```

## 3. Install the pre-commit hooks

Run the following command to install the pre-commit hooks defined in your configuration file:

```bash
pre-commit install
```

## 4. Run once on all files

To run the pre-commit hooks on all files in your repository, use the following command:

```bash
pre-commit run --all-files
```

!!! info
    After that, the hooks will automatically run on every commit.

## 5. Update the pre-commit hooks

To update the pre-commit hooks to the latest version, use the following command:

```bash
pre-commit autoupdate
``` 

!!! note
    You can also commit without running the pre-commit hooks by using the `--no-verify` flag:

    ```bash
    git commit --no-verify -m "your commit message"
    ```

