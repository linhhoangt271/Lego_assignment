# Model Versioning

Model implementations are consolidated in `Models_Summary.ipynb`. Use notebook section headings and markdown notes to document changes between model versions.

Keep large artifacts out of Git:

- checkpoints: `*.pth`, `*.pt`
- generated result folders: `*_results/`
- local dataset files: `images/`, `minifigs.json`

For important results, commit concise summaries in documentation rather than raw model outputs.
