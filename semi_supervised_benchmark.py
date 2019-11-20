import argparse
import os
import numpy as np
import torch
import SGC
import SGC.utils
import SGC.models
import SGC.metrics
import SGC.normalization

import pickle as pkl
import generate_graph
import sklearn
import sklearn.preprocessing

SGC.utils.set_seed(0, True)
random_state = np.random.RandomState(0)
use_cuda = True

datasets = ['STL',"flowers102",'ESC-50','IMDB']
dataset_default = datasets[0]
refined_path_default = "refined_datasets/features"
graph_path_default = os.path.join("graph","STL_Cosine_False_0_None.gz")
models = ["SGC","LabelProp","LogReg"]
model_default = "SGC"

import torch

class LabelPropagation(object):
    """Label propagation models from https://datascience.stackexchange.com/questions/45459/how-to-use-scikit-learn-label-propagation-on-graph-structured-data.
    
    Parameters
    ----------
    adj_matrix: torch.FloatTensor
        Adjacency matrix of the graph.
    """
    def __init__(self, adj_matrix,normalize = True):
        if normalize:
            self.norm_adj_matrix = self._normalize(adj_matrix)
        else:
            self.norm_adj_matrix = adj_matrix            
        if use_cuda:
            self.norm_adj_matrix = self.norm_adj_matrix.cuda()
        self.n_nodes = adj_matrix.size(0)
        self.one_hot_labels = None 
        self.n_classes = None
        self.labeled_mask = None
        self.predictions = None

    def _normalize(self,adj_matrix):
        """Computes D^-1 * W"""
        mask = torch.eye(adj_matrix.size(0), adj_matrix.size(0)).bool()
        if use_cuda:
            adj_matrix = adj_matrix.cuda()
            mask = mask.cuda()
#        adj_matrix.masked_fill_(mask, 1)
#        adj_matrix += torch.eye(adj_matrix.size(0), adj_matrix.size(0)).cuda()
        d = adj_matrix.sum(dim=1)
#        d[d == 0] = 1  # avoid division by 0 error
        d = torch.diag(torch.pow(d,-1/2))
        return torch.matmul(torch.matmul(d,adj_matrix),d)

    def _propagate(self):
        self.predictions = torch.matmul(self.norm_adj_matrix, self.predictions)

        # Put back already known labels
        self.predictions[self.labeled_mask] = self.one_hot_labels[self.labeled_mask]

    def _one_hot_encode(self, labels, idx_train):
        # Get the number of classes
        local_labels = torch.zeros_like(labels)-1
        if use_cuda:
            local_labels = local_labels.cuda()
        local_labels[idx_train] = labels[idx_train]
        classes = torch.unique(labels)
        self.n_classes = classes.size(0)

        # One-hot encode labeled data instances and zero rows corresponding to unlabeled instances
        unlabeled_mask = (local_labels == -1)
        local_labels[unlabeled_mask] = 0
        self.one_hot_labels = torch.zeros((self.n_nodes, self.n_classes), dtype=torch.float)
        if use_cuda:
            self.one_hot_labels = self.one_hot_labels.cuda()
        self.one_hot_labels = self.one_hot_labels.scatter(1, local_labels.unsqueeze(1), 1)
        self.one_hot_labels[unlabeled_mask, 0] = 0

        self.labeled_mask = ~unlabeled_mask

    def fit(self, labels, idx_train,max_iter=1000, tol=1e-3):
        """Fits a semi-supervised learning label propagation model.
        
        labels: torch.LongTensor
            Tensor of size n_nodes indicating the class number of each node.
            Unlabeled nodes are denoted with -1.
        max_iter: int
            Maximum number of iterations allowed.
        tol: float
            Convergence tolerance: threshold to consider the system at steady state.
        """
        with torch.no_grad():
            self._one_hot_encode(labels,idx_train)

            self.predictions = self.one_hot_labels.clone()
            prev_predictions = torch.zeros((self.n_nodes, self.n_classes), dtype=torch.float)
            if use_cuda:
                prev_predictions = prev_predictions.cuda()
            for i in range(max_iter):
                # Stop iterations if the system is considered at a steady state
                variation = torch.abs(self.predictions - prev_predictions).sum().item()

                if variation < tol:
                    #print(f"The method stopped after {i} iterations, variation={variation:.4f}.")
                    break

                prev_predictions = self.predictions
                self._propagate()

    def predict(self):
        return self.predictions

    def predict_classes(self):
        return self.predictions.max(dim=1).indices

def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    # From https://github.com/shchur/gnn-benchmark/blob/master/gnnbench/data/make_dataset.py
    
    num_samples, num_classes = len(labels), labels.max()+1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    # From https://github.com/shchur/gnn-benchmark/blob/master/gnnbench/data/make_dataset.py
    num_samples, num_classes = len(labels), labels.max()+1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=100, weight_decay=0,
                     lr=0.2):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = torch.nn.functional.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()

    return model

def test_regression(model, test_features, test_labels):
    model.eval()
    return SGC.metrics.accuracy(model(test_features), test_labels)


def run_semi_supervised_benchmark(dataset=dataset_default,refined_path=refined_path_default,graph_path=graph_path_default,model=model_default,minmaxscaler=False):

    if dataset == "STL":
        file = "stl.npz"
        nodes = 1000
    elif dataset == "ESC-50":
        file = "esc-50.npz"
        nodes = 2000
    elif dataset == "flowers102":
        file = "flowers102.npz"
        nodes = 1020
    file_path = os.path.join(refined_path_default,file)
    data = np.load(file_path)
    features = data["x"]
    labels = data["y"]
    graph = generate_graph.read_adjacence_matrix(nodes,graph_path)
   
    idx_train,idx_val,idx_test = get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=2, val_size=0)
    if minmaxscaler:
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        features = scaler.fit_transform(features)

    
    # porting to pytorch
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    #labels = torch.max(labels, dim=1)[1]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if use_cuda:
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()


    if model in ("LogReg","SGC"):
        if model == "LogReg":
            model = LogisticRegression(features.size(1),labels.max().item()+1)
            degree = 0
            if use_cuda:
                model.cuda()

        elif model == "SGC":
            degree = 2
            adj, features = SGC.utils.preprocess_citation(graph, features,normalization="None")
            adj = SGC.utils.sparse_mx_to_torch_sparse_tensor(adj).float()
            if use_cuda:
                adj = adj.cuda()
                features = features.cuda()


            model = SGC.models.get_model("SGC", features.size(1), labels.max().item()+1, 0, 0, use_cuda)

            features, _ = SGC.utils.sgc_precompute(features, adj, degree)

        model = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                                 100, 0, 0.001)
        acc_train = test_regression(model, features[idx_train], labels[idx_train]).item()*100
        acc_val = 0#test_regression(model, features[idx_val], labels[idx_val]).item()*100
        acc_test = test_regression(model, features[idx_test], labels[idx_test]).item()*100

    else:
        model = LabelPropagation(torch.FloatTensor(graph),False)
        model.fit(labels,idx_train)
        acc_train = SGC.metrics.accuracy(model.predict()[idx_train], labels[idx_train]).item()*100
        acc_val = 0#SGC.metrics.accuracy(model.predict()[idx_val], labels[idx_val]).item()*100
        acc_test = SGC.metrics.accuracy(model.predict()[idx_test], labels[idx_test]).item()*100
        
        
    return acc_train, acc_val, acc_test

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset',
                          choices=datasets, default=dataset_default,
                          help='Dataset with features extracted')
    parser.add_argument('--model',
                          choices=models, default=model_default,
                          help='Dataset with features extracted')
    
    parser.add_argument('--graph_path',
                          type=str, default=graph_path_default,
                          help='Path to the graph file to use')
    parser.add_argument('--refined_path',
                          type=str, default=refined_path_default,
                          help='Refined dataset path')

    args = parser.parse_args()
    
    acc_train, acc_val, acc_test = run_semi_supervised_benchmark(dataset=args.dataset,graph_path=args.graph_path,refined_path=args.refined_path,model=args.model)
    print("Train Accuracy: {:.2f} Validation Accuracy: {:.2f} Test Accuracy: {:.2f}".format(acc_train,acc_val, acc_test))
