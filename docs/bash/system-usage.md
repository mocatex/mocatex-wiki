# Set scripts for system wide use

With the follwowing steps you can setup a location where you can store all you bash scripts and then use them from everywhere in your system via the terminal.
The steps can be used for macOS and Linux systems.

## 1. Create a directory for your scripts

```bash
mkdir -p ~/.local/bin
```

## 2. Add the directory to your PATH

You can add the directory to your PATH by adding the following line to your `~/.bashrc`, `~/.bash_profile`, or `~/.zshrc` file:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

you can also do that directly by running the following commands:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
exec $SHELL -l # reload the shell
```

## 3. Create a script file

Write some bash code in a file and save it in the `~/.local/bin` directory.

Let you creativity run wild :)

!!! tip
    1. start every script with a **shebang** line to specify the interpreter: `#!/usr/bin/env bash`
    2. You don't need to add the `.sh` extension.
 
## 4 Prepare the Script for execution

Make the script executable by running the following command:

```bash
chmod +x ~/.local/bin/your-script-name.sh
``` 

## 5. Use the script

All done! Now you can use your script from anywhere in your terminal by simply typing its name:

```bash
your-script-name
```
