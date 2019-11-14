import sklearn.covariance
import sklearn
import sklearn.cluster
import sklearn.preprocessing 
import sklearn.metrics
import scipy
import numpy as np
import os
import gzip

datasets = ['STL','AudioSet','IMDB']
dataset_default = datasets[0]
graph_types = ["Cosine",'RBF','Covariance',"GraphLasso"]
graph_type_default = graph_types[0]
nn_default = 0
refined_path_default = "refined_datasets/features"
normalizations = ["None","RandomWalk","BothSides"]
normalization_default = "None"

def save_adjacence_matrix(adjacence_matrix, file_path, precision=3):
    with gzip.open(file_path, 'wb') as f:
        for idx in range(adjacence_matrix.shape[0]):
            line = adjacence_matrix[idx,:]
            values = np.where(line != 0)[0]
            for idx2 in values:
                value = line[idx2]
                if precision:
                    value = np.round(value,precision)
                write_string = "{}\t{}\t{}\n".format(idx,idx2,value)
                f.write(str.encode(write_string))

def read_adjacence_matrix(nodes, file_path):
    adj_matrix = np.zeros((nodes,nodes))
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            splitted = line.replace("\n","").split("\t")
            adj_matrix[int(splitted[0]),int(splitted[1])] = float(splitted[2])
    return adj_matrix
    


def largest_connected_components(adj_matrix, n_components=1):
    """Select the largest connected components in the graph. Code from 
    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.
    Returns
    -------
    sparse_graph : SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.
    """
    _, component_indices = scipy.sparse.csgraph.connected_components(adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return nodes_to_keep

def covariance(matrix):
    f = np.ones(matrix.shape[1])
    a = np.ones(matrix.shape[1])

    m = matrix.copy()
    w = f * a
    v1 = np.sum(w)
    v2 = np.sum(a)
    m -= np.sum(m * w, axis=1, keepdims=True) / v1
    _cov = np.dot(m * w, m.T) * v1 / (v1**2)    
    return _cov
    
def create_knnadjacence_matrix(matrix,k=4,symmetric = True):
    adjacence_matrix,weighted_adjacence_matrix = knn_over_matrix(matrix,k=k)
    if symmetric:
        symmetric_adj = force_symmetry(adjacence_matrix)
        return symmetric_adj, matrix*symmetric_adj
    else:
        return adjacence_matrix, weighted_adjacence_matrix
        
    
def knn_over_matrix(matrix,k=4):
    temp = np.argsort(-matrix,axis=1)[:,k-1] # Get K biggest index
    thresholds = matrix[np.arange(matrix.shape[0]),temp].reshape(-1,1) # Transform matrix into a column matrix of maximums
    adjacence_matrix = (matrix >= thresholds)*1.0 # Create adjacence_matrix
    weighted_adjacence_matrix = adjacence_matrix * matrix # Create weigthed adjacence_matrix
    return adjacence_matrix, weighted_adjacence_matrix

def force_symmetry(matrix):
    return np.minimum(matrix+matrix.T,1)

def generate_graph(dataset=dataset_default,graph_type=graph_type_default,minmaxscaler=False,nn=nn_default,refined_path=refined_path_default,normalization=normalization_default):

    if dataset == "STL":
        file = "stl.npz"
    file_path = os.path.join(refined_path_default,file)
    data = np.load(file_path)
    features = data["x"]
    labels = data["y"]
    if minmaxscaler:
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        index = 1
        index_max = features[index].argmax()
        features = scaler.fit_transform(features)
    if graph_type == "Covariance":
        graph = covariance(features)
    elif graph_type == "Cosine":
        graph = 1-scipy.spatial.distance.cdist(features,features,'cosine')
    elif graph_type == "RBF":
        distances = scipy.spatial.distance.cdist(features,features,'sqeuclidean')
        variance = np.var(distances)
        graph = np.exp(-distances/(2*variance))
    elif graph_type == "GraphLasso":
        graph_lasso = sklearn.covariance.GraphicalLasso(max_iter=1000)
        cov = graph_lasso.fit(features.T)
        graph = np.around(cov.covariance_,decimals=3)
    else:
        raise Exception("Graph type {} is not coded".format(args.type))

    if args.nn > 0:
        graph = create_knnadjacence_matrix(graph,args.nn)[1]

    if normalization == "RandomWalk":
        np.fill_diagonal(graph, 0)
        d = np.sum(graph, 1)
        d = np.power(d,-1)
        d = np.diag(d)
        graph = np.dot(d,graph)
    elif normalization == "BothSides":
        np.fill_diagonal(graph, 0)
        d = np.sum(graph, 1)
        d = np.power(d,-1/2)
        d = np.diag(d)
        graph = np.dot(np.dot(d,graph),d)
    save_path = os.path.join("graph","{}_{}_{}_{}_{}.gz".format(dataset,graph_type,minmaxscaler,nn,normalization))
    save_adjacence_matrix(graph,save_path)
    return graph

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset',
                          choices=datasets, default=dataset_default,
                          help='Dataset with features extracted')
    parser.add_argument('--graph_type',
                          choices=graph_types, default=graph_type_default,
                          help='Graph type to generate')
    parser.add_argument('--minmaxscaler',
                          action="store_true", default=False,
                          help='Use a min max scaler for the features')
    parser.add_argument('--refined_path',
                          type=str, default=refined_path_default,
                          help='Refined dataset path')
    parser.add_argument('--normalization',
                          choices=normalizations, default=normalization_default,
                          help='Adjacency matrix normalization')

    parser.add_argument('--nn',
                          type=int, default=nn_default,
                          help='Threshold n nearest neighbors and then symmetrize')


    args = parser.parse_args()
    
    generate_graph(dataset=args.dataset,graph_type=args.graph_type,minmaxscaler=args.minmaxscaler,refined_path=args.refined_path,normalization=args.normalization,nn=args.nn)
