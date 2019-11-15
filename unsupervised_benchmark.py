import sklearn
import sklearn.cluster
import sklearn.metrics
import numpy as np
import generate_graph
import os

datasets = ['STL','ESC-50','IMDB']
dataset_default = datasets[0]
refined_path_default = "refined_datasets/features"
graph_path_default = os.path.join("graph","STL_Cosine_False_0_None.gz")

def run_unsupervised_benchmark(dataset=dataset_default,graph_path=graph_path_default,refined_path=refined_path_default):
    if dataset == "STL":
        file = "stl.npz"
        nodes = 1000
        n_clusters = 10
    elif dataset == "ESC-50":
        file = "esc-50.npz"
        nodes = 2000
        n_clusters = 50
    file_path = os.path.join(refined_path_default,file)
    data = np.load(file_path)

    labels = data["y"]
    graph = generate_graph.read_adjacence_matrix(nodes,graph_path)
    clustering = sklearn.cluster.SpectralClustering(n_clusters=n_clusters,assign_labels="discretize",n_init=1000,random_state=0,affinity="precomputed")
    labels_result = clustering.fit_predict(graph)
    AMI = 100*sklearn.metrics.adjusted_mutual_info_score(labels, labels_result,average_method="arithmetic")
    NMI = 100*sklearn.metrics.normalized_mutual_info_score(labels, labels_result,average_method="arithmetic")
    print("AMI: {:.2f}, NMI: {:.2f}".format(AMI,NMI))
    return AMI, NMI

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
    
    run_unsupervised_benchmark(dataset=args.dataset,graph_path=args.graph_path,refined_path=args.refined_path)
