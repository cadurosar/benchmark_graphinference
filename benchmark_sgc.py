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
            acc_train, acc_train_std, acc_test, acc_test_std = 0,0,0,0
            
            try:
                if minmaxscaler == "False":
                    acc_train, acc_train_std, acc_test, acc_test_std = semi_supervised_benchmark.run_semi_supervised_benchmark(dataset=dataset,graph_path=os.path.join(graph_path_default,file),minmaxscaler=False,runs=100,split=5)
                else:
                    acc_train, acc_train_std, acc_test, acc_test_std = semi_supervised_benchmark.run_semi_supervised_benchmark(dataset=dataset,graph_path=os.path.join(graph_path_default,file),minmaxscaler=True,runs=100,split=5)                    
            except:
                print("Error semi supervised {}".format(file))
            graph_dict["acc_train"] = acc_train
            graph_dict["acc_train_std"] = acc_train_std
            graph_dict["acc_test"] = acc_test
            graph_dict["acc_test_std"] = acc_test_std
            all_dicts.append(graph_dict)
        except:
            print(file)
            continue
        df = pd.DataFrame(all_dicts)
        df.to_csv("results/sgc.csv",index=False)
#    print(df)
if __name__ == "__main__":

    benchmark_all()
