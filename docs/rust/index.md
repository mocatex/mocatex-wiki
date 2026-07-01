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

