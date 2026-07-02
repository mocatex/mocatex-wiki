---
title: "Rust"
---

# the Rust Programming Language

![Rust Banner](./assets/rust-banner.avif)

> Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety. It is designed to be a safe, concurrent, and practical language that supports functional and imperative-procedural paradigms.

## System Setup

1. Installing command-line-tools (for linker): `xcode-select --install`
2. Install `rustup` (Rust toolchain installer): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` -> pick the default installation option
3. test installation: `rustc --version`, `cargo --version` and `rustup --version` should all return a version number <br/>
-> maybe restart terminal first: `exec zsh -l`

Done! Rust is now system-wide installed and ready to use.

## IDE Setup

VSCode for now, others maybe follow...

### VSCode

After installing it from [the official website](https://code.visualstudio.com/), install the following important extensions:

- **rust-analyzer**: official Rust language server, with features like code completion, diagnostics, etc.
- **CodeLLDB**: for debugging Rust code
- **Even Better TOML**: for syntax highlighting and formatting of `Cargo.toml` files
- **Dependi**: shows outdated crates inline
- **Error Lens**: shows errors inline

## Project Setup

We can create a new Rust project using Cargo, the Rust package manager and build system. To create a new project, run the following command in your terminal:

```bash
cargo new my_project_name
```

or if we are already in a directory and want to initialize it as a Rust project:

```bash
cargo init
```

This will create the following directory structure:

```bash
my_project_name/
├── Cargo.toml
├── .gitignore
└── src/
    └── main.rs
```

!!! tip "Exercises"
    When you are learning Rust, you are probably going to have **many small exercises**. So we don't have to create a new project for each exercise, we can create a **`bin/` directory** in our `src/` directory and put each exercise in its own file. This way Rust allows **multiple `main()` functions** in the same project, as long as they are in different files.

Then we can run each exercise using the following command:

```bash
cargo run --bin exercise_name
```

If we want to run the default `main.rs` file, we can simply run:

```bash
cargo run
```

This run is **unoptimized** and is meant for development. If we want to run an **optimized** version of our code, we can use the following command:

```bash
cargo run --release
```
