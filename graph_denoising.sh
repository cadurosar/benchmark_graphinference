for dataset in toronto
do
    for nn in 5 10 20 30 40 50 100 200 500 1000 1500 2000
    do
        for normalization in None BothSides
        do
            for graph_type in RBF
            do
                time python generate_graph.py --dataset $dataset --normalization $normalization --graph_type $graph_type --nn $nn --minmaxscaler
                time python generate_graph.py --dataset $dataset --normalization $normalization --graph_type $graph_type --nn $nn
                time python generate_graph.py --dataset $dataset --normalization $normalization --graph_type $graph_type --nn $nn --minmaxscaler --add_self_loops
                time python generate_graph.py --dataset $dataset --normalization $normalization --graph_type $graph_type --nn $nn --add_self_loops

            done
        done
        
    done
done

for dataset in toronto
do
    for nn in 5 10 20 30 40 50 100 200 500 1000 1500 2000
    do
        for normalization in None BothSides
        do
            for graph_type in RBF L2Distance
            do
                time python generate_graph.py --dataset $dataset --normalization $normalization --graph_type $graph_type --nn $nn --minmaxscaler --nnk
                time python generate_graph.py --dataset $dataset --normalization $normalization --graph_type $graph_type --nn $nn --nnk
                time python generate_graph.py --dataset $dataset --normalization $normalization --graph_type $graph_type --nn $nn --minmaxscaler --add_self_loops --nnk
                time python generate_graph.py --dataset $dataset --normalization $normalization --graph_type $graph_type --nn $nn --add_self_loops --nnk
            done
        done
        
    done
done
