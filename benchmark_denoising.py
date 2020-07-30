import pandas as pd
import unsupervised_benchmark
import semi_supervised_benchmark
import os
import tqdm
import graph_denoising
home = os.path.expanduser("~")

graph_path_default = os.path.join("graph")

def benchmark_all():
    all_graphs = [f for f in os.listdir(graph_path_default) if "toronto" in f and os.path.isfile(os.path.join(graph_path_default, f))]
    all_dicts = list()
    for file in tqdm.tqdm(reversed(sorted(all_graphs)),total=len(all_graphs)):
        splitted = file.replace(".gz","").split("_")
        dataset = splitted[0]
        if dataset != "toronto":
            continue
        graph_type = splitted[1]
        minmaxscaler = splitted[2]
        nn = splitted[3]
        normalization = splitted[4]
        nnk = splitted[5]
        self_loop=self_loop = splitted[6]
        kalofolias = splitted[7]
        try:
            graph_dict = dict(dataset=dataset,graph_type=graph_type,minmaxscaler=minmaxscaler,nn=nn,normalization=normalization,nnk=nnk,kalofolias=kalofolias,self_loop=self_loop)
            snr, threshold = graph_denoising.run_graph_denoising(dataset=dataset,graph_path=os.path.join(graph_path_default,file))
            graph_dict["snr"] = snr
            graph_dict["threshold"] = threshold
            all_dicts.append(graph_dict)
            df = pd.DataFrame(all_dicts)
            df.to_csv("results/benchmark_denoising.csv",index=False)
        except:
            print(file)
            continue
#    print(df)
if __name__ == "__main__":

    benchmark_all()
