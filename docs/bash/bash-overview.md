# <center>Bash Overview</center>

- [Bash Overview](#bash-overview)
  - [1. General Structure of a Bash Script](#1-general-structure-of-a-bash-script)
  - [2. Bash Crash Course](#2-bash-crash-course)
    - [2.1 printing to the console](#21-printing-to-the-console)
    - [2.2 Variables](#22-variables)
      - [2.2.1 Environment Variables](#221-environment-variables)
        - [Predefined Environment Variables](#predefined-environment-variables)
        - [Defining custom Environment Variables](#defining-custom-environment-variables)
      - [2.2.2 Special/Positional Variables](#222-specialpositional-variables)
    - [2.3 Math Operations](#23-math-operations)
    - [2.4 Conditional Statements](#24-conditional-statements)
      - [2.4.1 Basic Structure](#241-basic-structure)
      - [2.4.2 Arithmetic Comparisons](#242-arithmetic-comparisons)
      - [2.4.3 Test comparisons](#243-test-comparisons)


!!! info
    BASH (Bourne Again SHell) is a command-line interpreter that is widely used in Linux and macOS systems. It allows users to execute commands, run scripts, and automate tasks.

## 1. General Structure of a Bash Script

A Bash script is a text file containing a series of commands that the Bash interpreter can execute.
The general structure includes:

- **Shebang Line**: The first line of the script, starting with `#!`, specifies the interpreter to be used. For Bash scripts, it is typically `#!/usr/bin/env bash`.
Here `env` finds the Bash interpreter automatically in the user's PATH. -> works accross different systems.
- **Safety Harness**: with the line `set -Eeuo pipefail` you can enable strict error handling in your script.
    - `-E`: ERR traps also get triggered in functions, command substitutions, and subshells.
    - `-e`: Exit immediately if a command exits with a non-zero status.
    - `-u`: error if you use an unset variable.
    - `-o pipefail`: in a pipeline, fail if **any** command fails (not just the last one).
- **Safe Filename Handling**: With `IFS=$'\n\t'` you can set the *Internal Field Separator* (IFS) to newline and tab only ('\n\t'), which helps in handling filenames with spaces.

So a good/safe starting point for a bash script would be:

```bash
#!/usr/bin/env bash

# -------- setup system --------
set -Eeuo pipefail
IFS=$'\n\t'
# ------------------------------
```

## 2. Bash Crash Course

> This is based on this [video playlist](https://youtube.com/playlist?list=PLT98CRl2KxKGj-VKtApD8-zCqSaN2mD4w&si=TxCQghs0sz4IHpoa)

### 2.1 printing to the console

- `echo`: prints text to the console with a newline `\n` at the end.
- `printf`: prints formatted text to the console without a newline at the end. More on the formatting [here](#21-printing-to-the-console).

### 2.2 Variables

- **define variable**: `var_name="Moritz"` (no spaces around the `=`).
- **access variable**: `$var_name` or `${var_name}`
-> curly braces are useful for expandning variable. (`${var_name}_suffix`)
- **command substitution**: `var_name=$(command)` runs the command in a subshell and assigns its output to the variable.

#### 2.2.1 Environment Variables

Environment variables are global variables that are available to all (sub)processes and shells.
They are typically defined in uppercase letters.

##### Predefined Environment Variables

*System Information:*

- `$HOSTNAME`: The name of the current host/computer.
- `$OSTYPE`: The operating system type.

*User Information:*

- `$HOME`: The home directory of the current user.
- `$USER`: The username of the current user.
- `$SHELL`: The path to the current user's shell.

*Location and Path:*

- `$PWD`: The current working directory.
- `$PATH`: A colon-separated list of directories that the shell searches for executable files.
- `$TMPDIR`: The directory for temporary files (usually `/tmp` on Unix-like systems).

*Script Utilities:*

- `$RANDOM`: A random integer between 0 and 32767.
- `$SECONDS`: The number of seconds since the shell was started.

##### Defining custom Environment Variables

- `export VAR_NAME="value"`: This command sets an environment variable named `VAR_NAME` with the value `"value"` and makes it available to all subprocesses.

#### 2.2.2 Special/Positional Variables

Special variables are predefined by the shell and have specific meanings.

- `$0`: The name of the script itself.
- `$1`, `$2`, ...: The first, second, etc. command-line arguments passed to the script.
- `$#`: The number of command-line arguments passed to the script.
- `$@`: All command-line arguments passed to the script as separate words.
- `$?`: The exit status of the last executed command (0 if successful, non-zero if an error occurred).
- `$$`: The process ID (PID) of the current shell.
- `$LINENO`: The current line number in the script.

### 2.3 Math Operations

- `(( expression ))`: Used for arithmetic operations. Inside the double parentheses, you can use standard arithmetic operators like `+`, `-`, `*`, `/`, and `%`.
- With a `$` in front of it, you can assign the result to a variable: `result=$(( 5 + 3 ))`
- `let`: It allows you to use standard arithmetic operators **without** the need for double parentheses. Example: `let result=5+3`
  
!!! info Good to know
    - You can use `++` and `--` for incrementing and decrementing variables.
    - You can use `+=`, `-=`, `*=`, `/=`, and `%=` for compound assignments.
    - You don't need to use `$` when referencing variables inside `(( ))` or with `let`.
    - Bash only supports integer arithmetic. For floating-point calculations, you can use tools like `bc` or `awk`.
    - `expr`: is a old way to perform arithmetic operations in bash. Only needed for very old bash versions.

### 2.4 Conditional Statements

Conditional statements allow you to execute different blocks of code based on certain conditions.

- `if`, `then`, `elif`, `else`, `fi`: Basic structure of an if statement in bash.

#### 2.4.1 Basic Structure

```bash
if [ condition ]; then
    # commands to execute if condition is true
elif [ another_condition ]; then
    # commands to execute if another_condition is true
    # you can have multiple elif blocks
else
    # commands to execute if none of the above conditions are true
fi
```

#### 2.4.2 Arithmetic Comparisons

When you only rely on integer values, you can use `(( ))` for comparisons:

```bash
if (( a < b )); then
```

Operators: `<`, `>`, `<=`, `>=`, `==`, `!=`

#### 2.4.3 Test comparisons

For string comparisons and more complex conditions, you can use the `test` command or `[ ]` brackets:

```bash
if [ "$str1" = "$str2" ]; then
```

- **Operators for Strings**: `=`, `!=`, `<`, `>`, `-z` (string is empty), `-n` (string is not empty), `=*` / `!=*` (case-insensitive match)
- **File Operators**: `-e` (file exists), `-f` (is a regular file), `-d` (is a directory), `-r` (is readable), `-w` (is writable), `-x` (is executable)
- **Combining Conditions**: Use `[[ ]]` for more complex conditions with `&&` (and) and `||` (or) and regex matching `=~`.
  
> use `!` for negation.

```bash
if [[ "$str1" = "hello" && -f "$file" && "$str2" =~ ^[0-9]+$ ]]; then
# true when str1 is "hello", file exists and str2 contains only digits
```

!!! tip
    - Always put spaces around the brackets and operators.
    - Use double quotes around variables to prevent issues with spaces or special characters.
    - When using `[[ ]]`, you don't need to quote variables for string comparisons, but it's still a good practice to do so.
