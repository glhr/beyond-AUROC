# beyond-AUROC
official code repository for the paper "Beyond AUROC &amp; co. for evaluating out-of-distribution detection performance"

## Basic usage

```python
# Generate scores for ID and OOD samples
id_data = np.random.normal(0,0.2,500)
ood_data = np.random.normal(0.4,0.1,500)

# Compute OOD metrics and plot histogram + threshold curve for AUTC
plot_ood_scores(id_data,ood_data)
```
## output
![Normalized histogram of OOD scores (left), FNR / FPR vs. threshold curves (right)](example_plots.png)

```python
# standard metrics & thresholds
{'aupr-in': 0.9255299568160484,
 'aupr-out': 0.95080956515798,
 'auroc': 0.9427345454545454,
 'fnr@95tnr': 0.35432499454644867,
 'fpr@95tpr': 0.23636363636363636,
 'thresh_95tnr': 0.35657367889311314,
 'thresh_95tpr': 0.23418827631646175}
 
# ours
auFNR 0.6044, auFPR 0.1528
--> AUTC 0.3786
```

## Synthetic examples

The [Jupyter notebook](./imaginary%20models.ipynb) contains the code for reproducing the visualizations and OOD performance of the imaginary models in the paper (Figs. 1, 3, 4, 6, 7).
