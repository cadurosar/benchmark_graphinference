import pandas as pd
import unsupervised_benchmark
import semi_supervised_benchmark
import os
import tqdm
home = os.path.expanduser("~")

graph_path_default = os.path.join(home,"benchmark_graphinference","graph")

def benchmark_all():
    all_graphs = [f for f in os.listdir(graph_path_default) if os.path.isfile(os.path.join(graph_path_default, f))]
    all_dicts = list()
    a = 0
    for file in tqdm.tqdm(reversed(sorted(all_graphs)),total=len(all_graphs)):
        try:
            splitted = file.replace(".gz","").split("_")
            dataset = splitted[0]
            graph_type = splitted[1]
            minmaxscaler = splitted[2]
            nn = splitted[3]
            normalization = splitted[4]
            graph_dict = dict(dataset=dataset,graph_type=graph_type,minmaxscaler=minmaxscaler,nn=nn,normalization=normalization)
            ami,nmi = unsupervised_benchmark.run_unsupervised_benchmark(dataset=dataset,graph_path=os.path.join(graph_path_default,file),assign_labels="discretize")
            acc_train, acc_val, acc_test = 0,0,0
            label_prop_train, label_prop_val, label_prop_test = 0,0,0
            
            if True or normalization == "None":
                label_prop_train, label_prop_val, label_prop_test = semi_supervised_benchmark.run_semi_supervised_benchmark(model="LabelProp",dataset=dataset,graph_path=os.path.join(graph_path_default,file),minmaxscaler=False)
                if minmaxscaler == "False":
                    acc_train, acc_val, acc_test = semi_supervised_benchmark.run_semi_supervised_benchmark(dataset=dataset,graph_path=os.path.join(graph_path_default,file),minmaxscaler=False)
                else:
                    acc_train, acc_val, acc_test = semi_supervised_benchmark.run_semi_supervised_benchmark(dataset=dataset,graph_path=os.path.join(graph_path_default,file),minmaxscaler=True)                    
            graph_dict["ami"] = ami
            graph_dict["nmi"] = nmi
            graph_dict["acc_train"] = acc_train
            graph_dict["acc_val"] = acc_val
            graph_dict["acc_test"] = acc_test
            graph_dict["label_prop_train"] = label_prop_train
            graph_dict["label_prop_val"] = label_prop_val
            graph_dict["label_prop_test"] = label_prop_test
            print(graph_dict)
            all_dicts.append(graph_dict)
        except:
            print(file)
            continue
    df = pd.DataFrame(all_dicts)
    df.to_csv("all_benchmark.csv",index=False)
    print(df)
if __name__ == "__main__":

    benchmark_all()