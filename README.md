# Zero-Shot Low-Rank Compression of Tabular Neural Networks via Marchenko-Pastur Tail-Quantile Spectral Thresholding

Supplementary code and results.

Authors: Rajendriya Wigar Punagi, Hasbi Yasin
Affiliation: Department of Statistics, Universitas Diponegoro, Semarang, Indonesia

## Contents

- `rancanganpublikasi.ipynb`: Kaggle notebook with the main experiment. The code cell with index 8 (zero-based; the 9th cell in the Kaggle interface) runs 4 datasets (Adult Income, Bank Marketing, Credit Default (Taiwan), HIGGS 98k) x 5 seeds x 2 baseline regimes (standard, regularized) for the baseline plus 18 compressed variants, and prints the accuracy, parameter, speedup and TOST tables used in the paper. Of the other code cells, only those with index 4 and 6 have saved outputs.
- `ablasi-pruning.ipynb`: Kaggle notebook for the bias-correction ablation (pre-activation and post-ReLU variants against the bias-free method). The models are retrained, so its baseline differs slightly from the baseline in the main tables. Labels in the notebook outputs and CSV files are in Indonesian; the CSV column `p_mcnemar` holds per-seed McNemar p-values and is not used in the paper.
- `ablasi_bias_per_seed.csv`: per-seed records of the bias-correction ablation.
- `ablasi_bias_selisih.csv`: paired differences against the bias-free variant.
- `figure1.py`: builds Figure 1 from `figure1_data.csv`. Run `python figure1.py --out Figure1.png`. Use `--no-jitter` to plot the exact speedups without horizontal offsets.
- `figure1_data.csv`: speedup (batch size 1024, seed 42, one thread), accuracy change (pp, mean of five seeds) and 95% CI, copied from the output of the cell with index 8 of `rancanganpublikasi.ipynb`.
- `Figure1.png`: Figure 1 as used in the paper. Proposed, EVR, uniform-rank, magnitude-pruning and small-model markers are shifted horizontally by fixed amounts of at most 3% of the axis width, identical in every panel, so that overlapping markers stay visible. Data values are unchanged.

## How to run

The notebooks were run on Kaggle (CPU only; `torch.device("cpu")`, latency measured with `torch.set_num_threads(1)`, training uses `torch.get_num_threads()`). Datasets are downloaded from OpenML with `sklearn.datasets.fetch_openml`, so internet access must be enabled.

1. Upload a notebook to Kaggle and run all cells with internet on. Set `QUICK = False` for the full experiment.
2. Python packages imported by the notebooks: numpy, pandas, torch, scikit-learn, scipy, matplotlib. Versions were not printed by the notebooks and are not recorded here.
3. Run time is about 67 minutes for cell 8 of `rancanganpublikasi.ipynb` and about 33 minutes for `ablasi-pruning.ipynb`, as recorded in the notebook metadata.

`figure1.py` needs only matplotlib.

## License

MIT License (see LICENSE).
