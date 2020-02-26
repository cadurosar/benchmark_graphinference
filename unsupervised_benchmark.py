import sklearn
import sklearn.cluster
import sklearn.metrics
import numpy as np
import generate_graph
import os

datasets = ['STL',"flowers102",'ESC-50','IMDB',"cora"]
dataset_default = datasets[0]
refined_path_default = os.path.join("refined_datasets","features")
graph_path_default = os.path.join("graph","STL_Cosine_False_0_None.gz")

def run_unsupervised_benchmark(dataset=dataset_default,graph_path=graph_path_default,refined_path=refined_path_default,assign_labels="kmeans"):
    if dataset == "STL":
        file = "stl.npz"
        nodes = 1000
        n_clusters = 10
    elif dataset == "ESC-50":
        file = "esc-50.npz"
        nodes = 2000
        n_clusters = 50
    elif dataset == "flowers102":
        file = "flowers102.npz"
        nodes = 1020
        n_clusters = 102
    elif dataset == "cora":
        file = "cora.npz"
        nodes = 2708
        n_clusters = 7
    file_path = os.path.join(refined_path,file)
    data = np.load(file_path,allow_pickle=True)

    labels = data["y"]
    graph = generate_graph.read_adjacence_matrix(nodes,graph_path)
    clustering = sklearn.cluster.SpectralClustering(n_clusters=n_clusters,assign_labels=assign_labels,n_init=1000,random_state=0,affinity="precomputed")
    labels_result = clustering.fit_predict(graph)
    AMI = 100*sklearn.metrics.adjusted_mutual_info_score(labels, labels_result,average_method="arithmetic")
    NMI = 100*sklearn.metrics.normalized_mutual_info_score(labels, labels_result,average_method="arithmetic")
    ARI = 100*sklearn.metrics.adjusted_rand_score(labels, labels_result)
    return AMI, NMI, ARI

def run_kmeans(dataset=dataset_default,graph_path=graph_path_default,refined_path=refined_path_default,assign_labels="kmeans"):
    if dataset == "STL":
        file = "stl.npz"
        nodes = 1000
        n_clusters = 10
    elif dataset == "ESC-50":
        file = "esc-50.npz"
        nodes = 2000
        n_clusters = 50
    elif dataset == "flowers102":
        file = "flowers102.npz"
        nodes = 1020
        n_clusters = 102
    elif dataset == "cora":
        file = "cora.npz"
        nodes = 2708
        n_clusters = 7
    file_path = os.path.join(refined_path,file)
    data = np.load(file_path)

    labels = data["y"]
    features = data["x"]
    clustering = sklearn.cluster.KMeans(n_clusters=n_clusters)
    labels_result = clustering.fit_predict(features)
    AMI = 100*sklearn.metrics.adjusted_mutual_info_score(labels, labels_result,average_method="arithmetic")
    NMI = 100*sklearn.metrics.normalized_mutual_info_score(labels, labels_result,average_method="arithmetic")
    ARI = 100*sklearn.metrics.adjusted_rand_score(labels, labels_result)
    return AMI, NMI, ARI


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
    
    AMI, NMI, ARI = run_kmeans(dataset=args.dataset,graph_path=args.graph_path,refined_path=args.refined_path)
    print("AMI: {:.2f}, NMI: {:.2f}, ARI: {:.2f}".format(AMI,NMI,ARI))
