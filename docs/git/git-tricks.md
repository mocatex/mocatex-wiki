# Git and GitHub Tricks

!!! info
    This page contains various tips and tricks for using Git and GitHub effectively. Whether you're a beginner or an experienced user, these tricks can help streamline your workflow and improve your productivity.

## Rename a GitHub Repository

### Step 0: Quick Backup

Not needed, but recommended!

```bash
git clone --mirror <repo-url>
```

This creates a bare clone of the repository, which includes all branches and tags.

### Step 1: Rename the Repository on GitHub

1. Go to the repository on GitHub.
2. Click on "Settings" -> "General".
3. Under "Repository name", type the new name for your repository.
4. Click "Rename".

!!! note
    GitHub automatically sets up redirects from the old repository name to the new one. Also GitHub Pages will be automatically updated to use the new repository name.

### Step 2: Update Local Repository

In your local repository, update the remote URL to point to the new repository name.

```bash
git remote set-url origin <new-repo-url>
```

### Step 3: Rename Local Directory (Optional)

If you want to rename your local directory to match the new repository name, you can do so with the following command:

```bash
mv old-repo-name new-repo-name
```

!!! attention
    Make sure to update any scripts or tools that reference the old repository name.

### Step 4: Verify Everything is Working

To ensure everything is working correctly, you can run the following commands:

```bash
git remote -v # to verify the remote URL
git fetch # to ensure you can fetch from the new remote
```
