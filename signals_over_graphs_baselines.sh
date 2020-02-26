for dataset in cora flowers102 ESC-50
do
    for nn in 0 5 10 20 30 40 50 100 200 500 1000
    do
        for normalization in None BothSides
        do
            for graph_type in Cosine RBF Covariance
            do
                time python generate_graph.py --dataset $dataset --normalization $normalization --graph_type $graph_type --nn $nn
            done
        done
        
    done
done
