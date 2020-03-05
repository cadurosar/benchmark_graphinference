import sklearn
import sklearn.cluster
import sklearn.metrics
import numpy as np
import generate_graph
import os
import pygsp

datasets = ["toronto"]
dataset_default = datasets[0]
refined_path_default = os.path.join("refined_datasets","features")
graph_path_default = os.path.join("graph","toronto_RBF_False_0_None.gz")

def compute_snr(noisy_signal,signal):
    diff = np.array(noisy_signal,dtype=np.float32)-np.array(signal,dtype=np.float32)
    SnR = 20*np.log10(np.linalg.norm(signal)/np.linalg.norm(diff))
    return SnR

def denoise(G,noisy_signal,threshold):
    g = pygsp.filters.Simoncelli(G,threshold)
    return g.filter(noisy_signal)[:,0]


def run_graph_denoising(dataset=dataset_default,graph_path=graph_path_default,refined_path=refined_path_default):
    if dataset == "toronto":
        file = "toronto.npz"
        nodes = 2202
    file_path = os.path.join(refined_path,file)
    data = np.load(file_path,allow_pickle=True)
    noisy_signal = data["x"]
    original_signal = data["y"].reshape(-1)
    graph = generate_graph.read_adjacence_matrix(nodes,graph_path)
    np.fill_diagonal(graph,0)
    G = pygsp.graphs.Graph(graph,lap_type="combinatorial")
    G.estimate_lmax()
    G.compute_fourier_basis()
    denoised_signals = list()
    test_threshold = list()
    for threshold in range(0,200,5):
        test_threshold.append(threshold/100)
    for threshold in test_threshold:
        denoised_signals.append(denoise(G,noisy_signal,threshold))
    all_snr = list()
    for denoised_signal in denoised_signals:
        all_snr.append(compute_snr(denoised_signal,original_signal))
    best_snr = np.max(all_snr)
    best_threshold = test_threshold[np.argmax(all_snr)]
    return best_snr, best_threshold

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset',
                          choices=datasets, default=dataset_default,
                          help='Dataset with features extracted')
    parser.add_argument('--graph_path',
                          type=str, default=graph_path_default,
                          help='Path to the graph file to use')
    parser.add_argument('--refined_path',
                          type=str, default=refined_path_default,
                          help='Refined dataset path')

    args = parser.parse_args()
    
    best_snr, best_threshold = run_graph_denoising(dataset=args.dataset,graph_path=args.graph_path,refined_path=args.refined_path)
    print("Best SnR: {:.2f}, Best Threshold: {:.2f}".format(best_snr,best_threshold))
