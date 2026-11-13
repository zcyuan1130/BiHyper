# BiHyper

Initial source release of BiHyper, a tail-aware two-sided hypergraph enhancement framework for recommendation.

## Included code

- `bihyper.py`: RecBole adapter, base encoder integration, training loss, and ablation switches.
- `bihyper_model.py`: factorized hypergraph propagation, gated residual enhancement, and conflict regularization.
- `hyper_utils.py`: user-side and item-side hypergraph construction.
- `evaluation.py`: intersectional group evaluation from recommendation lists or a RecBole model.

## Release status

This is an initial snapshot of the existing implementation, not a complete reproduction package. Training entry points, exact environment versions, dataset preparation, and final experiment configurations are not included yet. The source has not been validated end to end as a standalone installation.

The model depends on PyTorch, NumPy, SciPy, and RecBole. The adapter also attempts to load base models from RecBole-GNN before falling back to RecBole. Model files use relative imports and must be integrated into an appropriate Python package and registered with the training framework. Running `bihyper.py` directly is not a training command.

## File-based evaluation

With NumPy installed, the evaluation script accepts whitespace-separated integer IDs. Each line contains a user ID followed by item IDs. Recommendation item order is the ranking order. All files must use the same ID mapping.

```bash
python ndcg.py --train_file train.txt --test_file test.txt --rec_file recommendations.txt --k 10
```

The checkpoint-based evaluation mode additionally requires `predict.py`, which is not included in this snapshot. Use file-based evaluation until the missing integration is released.

## Implementation details to reconcile before reproducing paper results

- Current `hit@k` outputs are mean hit counts, not counts divided by `k`; interpret and convert them explicitly when comparing with normalized paper metrics.
- Disabling the gate currently returns the propagated representation directly, rather than using a fixed 0.5 mixing weight.
- Nodes tied at the degree quantile threshold are assigned differently by model tail selection and evaluation grouping. These boundary conventions need reconciliation.

These details document the current code; no experimental results or reproduction claims are implied by this release.

Datasets, checkpoints, paper drafts, third-party PDFs, and temporary authoring files are not included.
