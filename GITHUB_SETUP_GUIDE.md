# GitHub Setup & Permission Configuration

Follow these steps to push your repository and configure team access.

## Step 1: Create GitHub Repository (5 minutes)

### On GitHub Website:

1. Go to **https://github.com** and log in
2. Click the **+** icon (top right) → **New repository**
3. Fill in the details:
   - **Repository name**: `big-data-lego-classification` (or your choice)
   - **Description**: "LEGO minifigure classification with transfer learning"
   - **Visibility**: Select **Public** (for team collaboration)
   - **Initialize repository**: Leave **unchecked** (we have existing code)
4. Click **Create repository**

### You'll see instructions like:

```
…or push an existing repository from the command line

git remote add origin https://github.com/YOUR-USERNAME/big-data-lego-classification.git
git branch -M main
git push -u origin main
```

**Copy these commands** (we'll use them in Step 2)

---

## Step 2: Push Code to GitHub

In your terminal, navigate to the project and run:

```bash
cd /home/test/big_data_assignment_2

# Add GitHub as remote (REPLACE with your actual URL from Step 1)
git remote add origin https://github.com/YOUR-USERNAME/big-data-lego-classification.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Expected Output:
```
Enumerating objects: 32, done.
Counting objects: 100% (32/32), done.
Delta compression using up to X threads
Compressing objects: 100% (31/31), done.
Writing objects: 100% (32/32), 12.77 KiB | 2.55 MiB/s, done.
Total 32 (delta 5), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (5/5), done.
To github.com:YOUR-USERNAME/big-data-lego-classification.git
 * [new branch]      main -> main
Branch 'main' set up to track remote origin/main.
```

✅ **Your code is now on GitHub!**

---

## Step 3: Add Team Members (Collaborators)

### Add people with **Read & Write** access:

1. On GitHub, go to your repo
2. Click **Settings** (top menu)
3. Click **Collaborators** (left sidebar)
4. Click **Add people**
5. Type their **GitHub username** OR **email address**
6. Click **Select a collaborator** from dropdown
7. Choose permission level:
   - **Maintain** → Can push, review PRs, manage (recommended for team leads)
   - **Write** → Can push and review PRs (recommended for contributors)
   - **Read** → Can view and clone only (for reviewers/observers)
8. Click **Add [name] to [repo]**

### They will receive an invitation email

---

## Access Levels Explained

| Level | Push Code | Review PRs | Manage Settings | Best For |
|-------|-----------|-----------|-----------------|----------|
| **Write** | ✅ Yes | ✅ Yes | ❌ No | Active contributors |
| **Maintain** | ✅ Yes | ✅ Yes | ✅ Yes | Team leads |
| **Read** | ❌ No | ⚠️ Comment only | ❌ No | Reviewers/Observers |

---

## Step 4: Set Up Branch Protection (Recommended)

This prevents accidental pushes and ensures code review:

1. Go to **Settings** → **Branches**
2. Click **Add rule**
3. Under "Branch name pattern", type: `main`
4. Check these boxes:
   - ✅ **Require a pull request before merging**
   - ✅ **Require approvals** (set to 1 reviewer minimum)
   - ✅ **Dismiss stale pull request approvals**
5. Click **Create**

**Result**: All changes require a Pull Request and approval before merging to main.

---

## Step 5: Share Repository Link

Now share with your team:

```
Your repository: https://github.com/YOUR-USERNAME/big-data-lego-classification

Quick Start Instructions:
1. Clone: git clone <url>
2. Setup: pip install -r requirements.txt
3. Read: docs/SETUP.md
4. Run: python src/training/baseline_train.py
```

---

## Common Tasks After Setup

### Add a new team member later:

```
Settings → Collaborators → Add people
```

### Change someone's permission level:

```
Settings → Collaborators → Click their name → Change role
```

### Remove a team member:

```
Settings → Collaborators → Click their name → Remove
```

### View who has access:

```
Settings → Collaborators (see list)
```

---

## Troubleshooting

### "Permission denied (publickey)"

**Solution**: Set up SSH key on GitHub
1. Terminal: `ssh-keygen -t ed25519`
2. GitHub: Settings → SSH and GPG keys → New SSH key
3. Paste public key (from `cat ~/.ssh/id_ed25519.pub`)

Or use HTTPS instead:
```bash
git remote set-url origin https://github.com/YOUR-USERNAME/big-data-lego-classification.git
```

### "Repository already exists"

```bash
# Remove old remote
git remote remove origin

# Add correct remote
git remote add origin https://github.com/YOUR-USERNAME/big-data-lego-classification.git

# Push
git push -u origin main
```

### "Authentication failed"

```bash
# Create personal access token:
# 1. GitHub → Settings → Developer settings → Personal access tokens
# 2. Generate new token (select: repo, workflow)
# 3. Copy token
# 4. Use as password when pushing (will prompt)
```

---

## Team Workflow After Setup

Once repository is live:

1. **Team members clone**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/big-data-lego-classification.git
   ```

2. **They create feature branches**:
   ```bash
   git checkout -b feature/their-improvement
   ```

3. **They make changes and commit**:
   ```bash
   git commit -m "Clear description of changes"
   ```

4. **They push and create PR**:
   ```bash
   git push origin feature/their-improvement
   # Then create PR on GitHub
   ```

5. **You review and merge** (if branch protection enabled, requires approval)

---

## Verify Everything Works

After pushing:

1. ✅ Visit `github.com/YOUR-USERNAME/big-data-lego-classification`
2. ✅ See all files listed
3. ✅ See README.md rendered on homepage
4. ✅ See 2 commits in history
5. ✅ Can clone: `git clone <url>`
6. ✅ Collaborators invited and have access

---

## Summary

| Step | Action | Time |
|------|--------|------|
| 1 | Create GitHub repo | 2 min |
| 2 | Push code | 1 min |
| 3 | Add team members | 2 min |
| 4 | Set branch protection | 2 min |
| 5 | Share link | 1 min |
| **Total** | | **8 minutes** |

---

**Need help?** Refer to [GitHub's documentation](https://docs.github.com/en) or ask questions!
