---
icon: simple/haskell
---

# Haskell - Getting Started

![Haskell Logo](../assets/haskell-logo.avif){ .center }

> Haskell is a **purely functional programming language** known for its strong static typing, lazy evaluation, and powerful type system. The principles are pretty close to mathematical functions, which makes it a very logical and stable language to work with.

## Installation

To get everything set up, we need to install the following tools:

- **[GHC (Glasgow Haskell Compiler)](https://www.haskell.org/ghc/)** - The main compiler for Haskell.
- **Cabal** - A build system and package manager for Haskell.
- **HLS (Haskell Language Server)** - Provides IDE support for Haskell.

and for VS Code the official **[Haskell extension](https://marketplace.visualstudio.com/items?itemName=haskell.haskell)** -> Provides syntax highlighting and basic support for Haskell.

### Tools Installation

run the following command to install GHC, Cabal, and HLS:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
```

You can press `Enter` to accept the default options during the installation process, **except** when prompted to install HLS, make sure to select `Yes`! (No is here the default)

### Verify Installation

Now just quickly check if everything is installed correctly by running:

```bash
ghcup --version                          
ghc --version
ghci --version
cabal --version
haskell-language-server-wrapper --version
```

*Note:* on macOS, you might need to add the path to your shell configuration file (e.g., `.zshrc` or `.bash_profile`):

```bash
source ~/.ghcup/env 
```

## Haskell REPL - GHCi

Instead of writing and running entire programs from `.hs` files within a [project](#setup-a-haskell-project), you can also use the interactive REPL (Read-Eval-Print Loop) called GHCi to quickly test out Haskell code snippets and functions.

To start GHCi, simply run the following command in your terminal:

```bash
ghci
```

This will open the GHCi prompt, where you can enter Haskell expressions and see their results immediately. For example:

```haskell
Prelude> let add x y = x + y
Prelude> add 5 3
8
```

You can also load Haskell files into GHCi using the `:load` command:

```haskell
Prelude> :load MyModule.hs
```

If you are done you can exit GHCi by typing: `:quit` or simply `:q`.

## Setup a Haskell Project

Haskell Projects are a more structured and stable way to write and organize your Haskell code instead of using the GHCi REPL. They allow you to manage dependencies, build configurations, and organize your code into modules and libraries.

To create a new Haskell project, you can use Cabal. Run the following command in your terminal while being in the directory where you want to create your project:

```bash
cabal init -n --is-executable --package-name <your-package-name> --main-is Main.hs
```

This will give you the following structure:

```bash
.
├── CHANGELOG.md                -> keeps track of changes & updates to your project (optional)
├── app                         -> contains the main application code
│   └── Main.hs                 -> the entry point of your application
└── haskell-learning.cabal      -> the project configuration file
```

If you don't develop a library or an entire application, you can just delete the `CHANGELOG.md` file since it's not necessary for learning purposes.

Since we are just learning, and therefore probably create lots of different small exercises, we don't want to create a new project for each exercise. We have now two possible ways to **create and run multiple exercises** in the same project:

### run with `runghc`

For **quick testing** and running small Haskell scripts, you can simply create a new `.hs` file in the root directory of your project and run it with `runghc`. No configuration or setup is needed.

```bash
runghc Exercise1.hs
```

### run with configured `cabal`

The more stable yet a bit more complex way is to **create a new executable** for each exercise in the `cabal` file. This way you can run your exercises with `cabal run <executable-name>`.

To do this, you need to edit your `haskell-learning.cabal` file and add a new executable section for each exercise. Since we resue the same configs for each exercise, we can first create a common section for the shared configurations and then reference it in each executable section:

```cabal
common shared-exe
    import:           warnings
    hs-source-dirs:   app
    build-depends:    base ^>=4.18.3.0
    default-language: Haskell2010
```

Now we can reuse this common section in each executable section! We delete the default `executable <your-package-name>` section and add new ones for each exercise:

```cabal
executable <executable-name>
    import:           shared-exe
    main-is:          <exercise-file>.hs
```

Now we can run each exercise with:

```bash
cabal run <executable-name>
```

With that setup, you now have a clean and organized starting point for your Haskell learning journey!
