import pandas as pd
import unsupervised_benchmark
import semi_supervised_benchmark
import os
import tqdm
home = os.path.expanduser("~")

graph_path_default = os.path.join(home,"benchmark_graphinference","graph")

def benchmark_all():
    all_graphs = [f for f in os.listdir(graph_path_default) if "toronto" not in f and os.path.isfile(os.path.join(graph_path_default, f))]
    all_dicts = list()
    for file in tqdm.tqdm(reversed(sorted(all_graphs)),total=len(all_graphs)):
        try:
            splitted = file.replace(".gz","").split("_")
            dataset = splitted[0]
            graph_type = splitted[1]
            minmaxscaler = splitted[2]
            nn = splitted[3]
            normalization = splitted[4]
            nnk = splitted[5]
            kalofolias = splitted[6]
            
            graph_dict = dict(dataset=dataset,graph_type=graph_type,minmaxscaler=minmaxscaler,nn=nn,normalization=normalization,nnk=nnk,kalofolias=kalofolias)
            ami,nmi,ari = 0,0,0

            try:
                ami,nmi, ari = unsupervised_benchmark.run_unsupervised_benchmark(dataset=dataset,graph_path=os.path.join(graph_path_default,file),assign_labels="discretize")
            except:
                print("Error unsupervised {}".format(file))
            
            graph_dict["ami"] = ami
            graph_dict["ari"] = ari
            graph_dict["nmi"] = nmi
#            print(graph_dict)
            all_dicts.append(graph_dict)
        except:
            print(file)
            continue
        df = pd.DataFrame(all_dicts)
        df.to_csv("results/unsupervised.csv",index=False)
#    print(df)
if __name__ == "__main__":

    benchmark_all()
