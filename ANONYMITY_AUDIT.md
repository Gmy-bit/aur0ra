# Anonymity and packaging audit

This report covers the source files included in the cleaned archive. It does not cover a future GitHub remote, an existing Git history, checkpoints, datasets, training logs generated outside the source tree, or third-party hosting services.

## Changes made

- Replaced machine-specific absolute paths in training/evaluation launchers, output-directory helpers, reference-set scripts, NLTK setup, and evaluation-data preparation with repository-relative defaults or environment-variable overrides.
- Removed the machine-specific local evaluation configuration directory, bytecode caches, editor swap files, and an environment-specific wait-for-GPU launcher.
- Added `.gitignore` rules for datasets, checkpoints, run outputs, local environments, caches, NLTK data, and common secret files.
- Reworked the SugarCrepe wrapper to use the in-repository scene-understanding evaluator and aligned scene-understanding config paths with the preparation script.
- Added the AURORA-specific ViT-S/16 config `configs/train_hycoslip_vit_s.py` and launcher `scripts/run_train_hycoslip.sh`.
- The AURORA launcher defaults to `MODEL_NAME=hycoslip`, `USE_BOXES=true`, root-angle sigmoid alignment, `ENTAILMENT_LOSS_TYPE=radial_shadow` (RAOC/HEC branch), and `NORM_REG_TYPE=argent_space` with `NORM_WEIGHT=0.05`.
- Added an English README and a Chinese training/evaluation guide with the AURORA launch command, GRIT preprocessing, checkpoint resumption, and downstream evaluation commands.

## Validation performed

- Python source files passed syntax compilation.
- Shell scripts passed `bash -n`.
- The AURORA config passed Python AST/syntax checks, and its key model/RAOC/norm settings passed static assertions. A model-factory smoke test confirmed that `hycoslip` resolves to the MERU-family class and accepts `radial_shadow` plus `argent_space`. A full config-loader/runtime check was not possible in this environment because optional project dependencies such as `timm`, `hydra-core`, and `webdataset` are not installed.
- The AURORA shell launcher passed `bash -n`; a mocked-launcher smoke test confirmed that it forwards `MODEL_NAME=hycoslip`, `USE_BOXES=true`, `radial_shadow`, `argent_space`, the requested weights, batch size, GPU count, and iteration budget to the shared training entry point. No GPU training was run in this environment.
- A source scan found no matches for common personal absolute-path patterns (`/home/`, `/mnt/`, `/root/`, `/Users/`), Windows user paths, or email-like strings. This is a static scan, not a guarantee against metadata or newly generated artifacts.
- The supplied archive did not contain a `.git` directory; therefore no Git commit history was included in the cleaned package.

## Remaining release checks

- Create a fresh anonymous Git repository and do not push the original personal repository's Git history.
- Before publishing, inspect generated `config.yaml` files, logs, TensorBoard event files, notebooks, model cards, and external links; these can contain local paths or account identifiers even when the source tree is clean.
- Confirm the applicable upstream licenses and add the required `LICENSE`/`NOTICE` files. The supplied archive did not contain a root `LICENSE` file, while some source files retain upstream copyright headers.
- The ImageNet–WordNet assets required by hierarchical evaluation (`assets/imagenet_synset/all_synsets.pkl`, `all_ancestors_indices.pkl`, and `imagenet_isa.txt`) are not included and must be supplied separately.
