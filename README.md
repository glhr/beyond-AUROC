# beyond-AUROC
official code repository for the paper "Beyond AUROC &amp; co. for evaluating out-of-distribution detection performance"

## Basic usage

```python
# Generate scores for ID and OOD samples
id_data = np.random.normal(0,0.1,500)
ood_data = np.random.normal(0.9,0.5,500)

# Compute standard OOD metrics and plot histogram + threshold curve for AUTC
plot_ood_scores(id_data,ood_data)
```

## Synthetic examples

The Jupyter notebook "imaginary models" contains the code for reproducing the visualizations and OOD performance of our imaginary models (Figs. 1, 3, 4, 6, 7).
