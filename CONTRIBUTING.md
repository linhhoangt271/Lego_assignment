# Contributing

Keep model code in `Models_Summary.ipynb`. Do not add standalone training scripts unless the team decides to split the workflow again.

Before committing:

```bash
git status
python -m json.tool Models_Summary.ipynb > /dev/null
```

Do not commit local data, checkpoints, generated result folders, logs, or cache files. They are covered by `.gitignore`.
