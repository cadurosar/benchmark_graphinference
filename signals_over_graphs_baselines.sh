for dataset in cora flowers102 ESC-50
do
    for nn in 0 5 10 20 30 40 50 100 200 500 1000
    do
        for normalization in None BothSides
        do
            for graph_type in Cosine RBF Covariance
            do
                time python generate_graph.py --dataset $dataset --normalization $normalization --graph_type $graph_type --nn $nn
                time python generate_graph.py --dataset $dataset --normalization $normalization --graph_type $graph_type --nn $nn --minmaxscaler
            done
        done
        
    done
done

for dataset in cora flowers102 ESC-50
do
    for nn in 5 10 20 30 40 50 100 200 500 1000
    do
        for normalization in None BothSides
        do
            for graph_type in Cosine RBF Covariance L2Distance
            do
                time python generate_graph.py --dataset $dataset --normalization $normalization --graph_type $graph_type --nn $nn --nnk
                time python generate_graph.py --dataset $dataset --normalization $normalization --graph_type $graph_type --nn $nn --minmaxscaler --nnk
            done
        done
        
    done
done
