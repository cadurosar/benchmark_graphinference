# Graph topology inference benchmarks for machine learning

Code for benchmarking graph topology inference methods designed to improve performance of machine learning methods. We provide code for simple plug and play evaluation of new methods and also some baseline results.

## Datasets

We provide 4 datasets (cora, toronto, ESC-50 and ) in numpy and Matlab format. The files are available in the refined_datasets folder. We also provide code for reproducing our feature extraction by running the extract_features.py script. If you want to re-extract the features you will have to download the raw data using the .sh scripts in the download_raw (./download_raw/{dataset}.sh DATA_FOLDER) folder 

## Graph inference baselines

We provide code for generating both naive inference (KNN with different values of k/similarity kernels) and [NNK](https://arxiv.org/abs/1910.09383). We thank the authors of NNK for their easy to use and accesible [code](https://github.com/STAC-USC/PyNNK_graph_construction). We also test with the [Kalofolias inference method](https://openreview.net/forum?id=ryGkSo0qYm), but for the moment we do not provide the code we used to do so (because it is done in matlab/octave we would need to include other toolboxes).

Our methods accept graphs in plain/compressed(gz) text form. A function save_adjacence_matrix is available in generate_graph.py to convert from numpy to this representation. Each line of the file contains:

```
node_i node_j weight
```

## Testing

We provide three scripts, unsupervised_benchmark.py, semi_supervised_benchmark.py and graph_denoising.py that allow one to quickly benchmark their inferred graphs.

### Spectral clustering example

To run the spectral clustering test for a normalized Cosine 10-nn graph:

```
python unsupervised_benchmark.py --graph_path graph/cora_Cosine_False_10_BothSides_False_False.gz --dataset cora
```

result:

```
AMI: 20.98, NMI: 21.31, ARI: 9.51
```
### Label propagation example

To run the Label prop test for a normalized Cosine 10-nn graph:

```
python semi_supervised_benchmark.py --graph_path graph/cora_Cosine_False_10_BothSides_False_False.gz --dataset cora --model LabelProp
```

result:

```
Train Accuracy: 100.00, Test Accuracy: 67.79
Train STD: 0.00, Test STD: 1.08
```

### SGC example

To run the SGC test for a normalized Cosine 10-nn graph:

```
python semi_supervised_benchmark.py --graph_path graph/cora_Cosine_False_10_BothSides_False_False.gz --dataset cora --model SGC
```

result:

```
Train Accuracy: 80.47, Test Accuracy: 70.34
Train STD: 1.31, Test STD: 0.97
```

### Graph denoising example

To run the graph denoising using a fully connected RBF graph:

```
python graph_denoising.py --graph_path graph/toronto_RBF_False_0_BothSides_False_False.gz
```

result:

```
Best SnR: 10.20, Best Threshold: 1.25
```


## Reproducing the results from the paper

To reproduce the results of the paper, first you have to run the graph_denoising.sh and signals_over_graph_baselines.sh scripts. This will generate all the inferred graphs. You can then run benchmark_{task}.py to get the results for each task

### Baseline results

#### Task 1

AMI scores:

|        Method       | Inference/Dataset | ESC-50 | cora   | flowers102 |
|:-------------------:|:-----------------:|:------:|:------:|:----------:|
|       C-means       |                   |  0.59  | 0.10   |    0.36    |
| Spectral clustering |       Naive       |  0.65  |**0.34**|  **0.46**  |
| Spectral clustering |        NNK        |**0.66**|**0.34**|    0.44    |
| Spectral clustering |     Kalofolias    |  0.65  | 0.27   |    0.44    |

We also provide non aggregated results in results/unsupervised.csv

#### Task 2

Mean test accuracy +- Standard deviation


| Method              | Inference/Dataset | ESC-50                 | cora                   | flowers102             |
|---------------------|-------------------|------------------------|------------------------|------------------------|
| Logistic Regression | Inference/Dataset |      52.92% +- 1.9     |      46.84% +- 1.6     |      33.51% +- 1.7     |
|  Label Propagation  |       Naive       |      59.05% +- 1.8     |      58.86% +- 2.9     |      36.73% +- 1.6     |
|  Label Propagation  |        NNK        |      57.44% +- 2.2     |      58.66% +- 2.9     |      33.57% +- 1.6     |
|  Label Propagation  |     Kalofolias    |      59.16% +- 1.8     |      58.60% +- 3.4     |      37.01% +- 1.7     |
|         SGC         |       Naive       |      60.48% +- 2.0     |      67.19% +- 1.5     |      37.73% +- 1.5     |
|         SGC         |        NNK        |      61.38% +- 2.0     |      66.58% +- 1.5     |      36.81% +- 1.5     |
|         SGC         |     Kalofolias    |      59.36% +- 2.0     |      66.28% +- 1.5     |      37.5% +- 1.5      |

We also provide non aggregated results in results/sgc.csv and results/labelprop.csv

#### Task 3

| Graph        | Best SNR       |
|--------------|----------------|
| No denoising | 7              |
| Real graph   | 10.32          |
| Kalofolias   | **10.41**      |
| RBF NNK      | 9.99           |
| RBF KNN      | 9.80           |

We also provide non aggregated results in results/benchmark_denoising.csv
