# Repo conventions for Claude

## Workflow: feature branch + PR (no direct main pushes)

Direct pushes to `main` are not allowed. All work goes through a feature
branch and a pull request.

### After every `git push` to a feature branch

Immediately open (or update) a pull request via the GitHub MCP:

1. Check whether an open PR already exists for the head branch:
   `mcp__github__list_pull_requests` with
   `head=<owner>:<feature-branch>` and `state=open`.
2. If none exists, create one with `mcp__github__create_pull_request`
   targeting `base=main`. The PR title should summarize the change in <70
   chars; the body should follow the standard template (Summary + Test
   plan).
3. If a PR already exists, do not open a duplicate — just confirm it is
   still open and report its number to the user.

Never `git push origin main` directly, and never merge a PR without
explicit user confirmation. Auto-opening the PR is allowed; auto-merging
it is not.

## Data pipeline

The classification model used by the visualizer is a **linear probe on
ResNet101 features**, not the ImageNet-1k head:

- `scripts/pca_reduce.py` runs ResNet101 once, emitting both 2048-D
  features (`avg_pool` layer) and ImageNet-1k top-5 predictions.
- `scripts/train_tiny_classifier.py` then replaces the ImageNet-1k
  predictions in `data/classification_results.json` with cross-validated
  outputs from a logistic regression head trained on the actual classes
  present in `data/images/`.

Re-run order when regenerating data: `pca_reduce.py` first (needs
TensorFlow + ImageNet weights), then `train_tiny_classifier.py` (only
needs `numpy` + `scikit-learn`, reuses the features already on disk).

The UI consumes `class_id`, `class_name`, `probability`, `top5`,
`source_id`, `source_label`, `path`, `date`. Any new classifier must
preserve this schema or `app/app.js` will break.
