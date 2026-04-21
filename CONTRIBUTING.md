# Contributing Guide

This document explains how to collaborate on the LEGO minifigure classification project using Git.

## 🚀 Getting Started for Team Members

### 1. Clone the Repository

```bash
git clone <repo-url> big_data_assignment_2
cd big_data_assignment_2
```

### 2. Set Up Your Environment

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate    # Linux/Mac
# or
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

**Full setup guide**: See `docs/SETUP.md`

### 3. Download Data (if not already present)

```bash
# Images (2.1GB) and minifigs.json (11MB) should be in project root
# Ask team lead if you need the data

# Verify:
ls -lh images/ | head -5
head -20 minifigs.json
```

### 4. Run Quick Test

```bash
python src/training/baseline_train.py
# Should complete in 5-10 minutes
```

---

## 📋 Workflow: Making Changes

### Branch Naming Convention

Create descriptive branch names:

```bash
# New feature
git checkout -b feature/v4-vision-transformer

# Bug fix
git checkout -b fix/town-split-imbalance

# Experiment
git checkout -b experiment/ensemble-predictions

# Documentation
git checkout -b docs/update-training-guide
```

### Making Your First Commit

```bash
# 1. Make your changes
# Example: Improve model training
vim src/training/optionB_v3.py

# 2. Verify changes
git status
git diff src/training/optionB_v3.py

# 3. Stage changes
git add src/training/optionB_v3.py

# 4. Commit with clear message
git commit -m "Improve learning rate scheduling for V3 model

- Use exponential decay instead of cosine annealing
- Reduces early epoch oscillation
- Expected F1 improvement: +1-2%

Tested on fold 0 with baseline V3 params."

# 5. Push to remote
git push -u origin feature/better-lr-scheduling
```

### Commit Message Format

Follow this structure:

```
[Type]: [Short description under 70 chars]

[Detailed explanation - what and why, not how]
[Reference issue if applicable: closes #123]

[Optional: testing details]
[Optional: performance impact]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `improve`: Enhancement to existing feature
- `perf`: Performance optimization
- `refactor`: Code cleanup (no functional change)
- `docs`: Documentation updates
- `test`: New tests
- `chore`: Build, config, dependencies

**Examples:**
```
feat: Add Vision Transformer model variant

Implements ViT-Base as alternative to EfficientNet-B4.
Expected to improve F1 by 2-3% on large datasets.

Tested on 1-fold CV with SupCon loss.

fix: Handle missing town metadata in data loader

Previously crashed when minifigs.json missing 'town' field.
Now defaults to 'Unknown' and logs warning.

improve: Reduce memory usage by 30% with gradient checkpointing

Enables batch size 32 on 8GB GPUs (was 8 before).
Minimal speed impact (~5% slower per epoch).
```

---

## 🔄 Creating Pull Requests (PRs)

### Before Creating PR

```bash
# 1. Fetch latest main
git fetch origin
git rebase origin/master

# 2. Run tests locally
python src/training/baseline_train.py    # Sanity check
python -m pytest tests/                   # If tests exist

# 3. Update documentation
# If you changed training behavior, update:
# - docs/TRAINING_GUIDE.md (if new workflow)
# - docs/ARCHITECTURE.md (if model changes)
# - README.md (if major change)

# 4. Push to remote
git push origin feature/your-feature
```

### Opening PR on GitHub

```bash
# Via CLI:
gh pr create --title "Add Vision Transformer backbone" \
  --body "## Summary
- Implements ViT-Base for comparison with EfficientNet-B4
- Expected +2-3% F1 improvement

## Testing
- Tested on 1 fold with baseline V3 hyperparameters
- Validation F1: 0.7250 (baseline: 0.7207)

## Checklist
- [x] Code follows project style
- [x] Documentation updated
- [x] No CUDA/memory regressions
- [x] Results logged in results/"

# Via web UI:
# 1. Go to GitHub repo
# 2. Click "New Pull Request"
# 3. Select your branch
# 4. Fill in title and description
# 5. Request review from team members
```

### PR Checklist

Before requesting review:

- [ ] Branch is up to date with `master`: `git rebase origin/master`
- [ ] All commits have clear messages
- [ ] Code runs without errors: quick test passed
- [ ] No large files committed (> 100MB model checkpoints)
- [ ] Documentation updated if needed
- [ ] Results documented in commit message or PR description
- [ ] No sensitive data (API keys, passwords) in code

---

## 📊 Sharing Experimental Results

### After Training Completes

```bash
# 1. Document results in PR description
git log --oneline | head -5
# Add to PR: "Trained V3 with params X, achieved F1=0.7207"

# 2. Save key metrics
# Don't commit large checkpoint files, but DO commit:
# - results.txt (metrics)
# - config.json (hyperparameters)
# - figures: confusion_matrix.png, training_curves.png

# Example:
git add optionB_v3_results/classification_report.txt
git add optionB_v3_results/confusion_matrix.png
git commit -m "Log V3 SupCon results: F1=0.7207"

# 3. Create PR with results
git push origin feature/v3-supcon-results
```

### Results Documentation Template

Create a results section in your PR:

```markdown
## Results

### Configuration
- **Model**: EfficientNet-B4 (380px)
- **Loss**: SupCon + CrossEntropy
- **Hyperparameters**: LR=3e-4, batch=16, epochs=20
- **Hardware**: NVIDIA A100, 8GB VRAM

### Metrics
- **Validation F1**: 0.7207 ± 0.0167 (3-fold CV)
- **Best Single Fold**: 0.7432
- **Improvement over Baseline**: +9.4%

### Key Insights
- SupCon outperforms Focal by 0.4% F1
- Model converges in ~18 epochs
- No overfitting detected (val ≈ train)

### Confusion Analysis
- Best class: Star Wars (F1=0.91)
- Worst class: Generic (F1=0.35)
- Main confusion: Town ↔ Castle

### Files
- Best checkpoint: `optionB_v3_hpopt_results/best_model_focal_fold1.pth`
- Full logs: `optionB_v3_hpopt_results/cv_results.txt`
- Detailed analysis: View in TensorBoard
```

---

## 🛑 Common Mistakes to Avoid

### ❌ Committing Large Files

```bash
# DON'T DO THIS:
git add *.pth                    # Model checkpoints
git add best_model.pt            # Too large for git
git add images/                  # Data folder

# DO THIS INSTEAD:
echo "*.pth" >> .gitignore
echo "images/" >> .gitignore
echo "minifigs.json" >> .gitignore
git add .gitignore
git commit -m "Ignore large files"
```

### ❌ Pushing Incomplete Work

```bash
# DON'T do:
git push while experiment still running
git push untested code

# DO:
# 1. Let training finish
# 2. Verify code runs
# 3. Then push
```

### ❌ Force Pushing to Master

```bash
# NEVER DO:
git push -f origin master       # ❌ WILL BREAK FOR EVERYONE

# If you make a mistake on master:
git revert <commit-hash>        # ✅ Safe, creates new commit
git reset --soft HEAD~1         # ✅ Undo last commit, keep changes
```

### ❌ Merging Without Testing

```bash
# BEFORE merging PR:
# 1. Pull the branch: git checkout feature/xyz
# 2. Run quick test: python baseline_train.py
# 3. Check for conflicts: git diff master
# 4. Then merge via GitHub PR UI
```

---

## 📈 Collaboration Best Practices

### 1. Keep Branches Short-Lived

```bash
# Good: 1-2 days of work
git checkout -b feature/warmup-epochs
# ...make 3-5 commits...
# Create PR same day

# Bad: 2+ weeks of changes
# ❌ Creates merge conflicts, hard to review
```

### 2. Rebase Before Pushing

```bash
# Before push:
git fetch origin
git rebase origin/master        # Update with latest changes

# Handle any conflicts locally before pushing
```

### 3. Review Each Other's Code

```bash
# When PR is ready:
# 1. Request review from team member
# 2. They check:
#    - Does it run?
#    - Are results better/similar?
#    - Is code clear?
# 3. Approve or request changes
# 4. Merge when approved
```

### 4. Keep History Clean

```bash
# Before submitting PR:
git log --oneline origin/master..HEAD
# Should show clear, logical commits

# If messy, squash:
git rebase -i origin/master
# Mark some commits as 'squash' to combine them
```

---

## 🐛 Handling Merge Conflicts

### If You Get Conflicts

```bash
# 1. Try auto-merge
git merge origin/master

# 2. If conflicts occur, Git marks them:
# <<<<<<< HEAD
# your changes
# =======
# their changes
# >>>>>>> branch-name

# 3. Edit files to resolve (keep both? pick one?)
vim conflicted_file.py

# 4. After resolving:
git add conflicted_file.py
git commit -m "Resolve merge conflict with master"
git push origin feature/your-branch

# 5. PR will update automatically
```

---

## 📞 Getting Help

### Questions?

```bash
# 1. Check documentation first
cat README.md
cat docs/TRAINING_GUIDE.md
cat docs/ARCHITECTURE.md

# 2. Check existing issues
gh issue list
gh issue view <number>

# 3. Ask in team chat or create issue
gh issue create --title "Question about ..."
```

### Want to Discuss a Change?

```bash
# Create a draft PR for discussion
gh pr create --draft --title "[WIP] New idea: feature X"

# Team can review and discuss before you complete
```

---

## 📚 Useful Git Commands

```bash
# View commit history
git log --oneline --graph --all

# See what changed
git diff master...your-branch

# Undo last commit (keep changes)
git reset --soft HEAD~1

# List all branches
git branch -a

# Delete local branch
git branch -d feature/done

# Sync fork with upstream
git fetch upstream
git rebase upstream/master

# Find who changed what
git blame optionB_v3.py
git log -p src/training/baseline_train.py

# Search git history
git log -S "SupCon" --oneline
```

---

## 🎯 Team Conventions

### Directory Structure

```
big_data_assignment_2/
├── src/
│   ├── training/          # Training scripts (main work here)
│   ├── models/            # Model definitions
│   ├── data/              # Data loading code
│   └── utils/             # Shared utilities
├── docs/                  # Documentation (edit when needed)
├── notebooks/             # Jupyter analysis (optional)
├── tests/                 # Tests (create if needed)
└── README.md
```

### Code Style

- Use 4-space indentation
- Keep lines under 100 characters (except URLs/data)
- Add docstrings to functions: `"""One-line summary."""`
- Use meaningful variable names

### Versioning Models

```bash
# Save like this:
best_model_v3_focal.pth       # Version_loss_variant
best_model_v4_vit.pth

# Log metadata:
# Save config.json alongside model
{
  "model": "EfficientNet-B4",
  "loss": "SupCon",
  "f1_score": 0.7207,
  "train_date": "2026-04-09"
}
```

---

**Last Updated**: 2026-04-21  
**Questions?** Check docs/ or create an issue on GitHub
