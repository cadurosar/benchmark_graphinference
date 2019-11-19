import pandas as pd
import unsupervised_benchmark
import os
import tqdm

def benchmark_all():
    all_graphs = [f for f in os.listdir("graph") if os.path.isfile(os.path.join("graph", f))]
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
            ami,nmi = unsupervised_benchmark.run_unsupervised_benchmark(dataset=dataset,graph_path=os.path.join("graph",file))
            graph_dict["ami"] = ami
            graph_dict["nmi"] = nmi
            print(graph_dict)
            all_dicts.append(graph_dict)
        except:
            continue
    df = pd.DataFrame(all_dicts)
    df.to_csv("unsupervised_benchmark.csv",index=False)
    print(df)
if __name__ == "__main__":

    benchmark_all()