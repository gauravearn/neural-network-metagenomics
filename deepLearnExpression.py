def deepLearnExpression(csv_file, fasta_file, control_columns, 
                                replicates_columns_1, 
                                      replicates_column_2,
                                              replicates_column3):
    """summary_line
    a function to generate the hidden layers from the given fasta
    and the expression files. it takes the replicate columns and then
    calculates the expression and length as a hidden layer. Applying to
    the transcriptomics, meta transcriptomics and other expression datasets
    
    Keyword arguments:
    argument -- description
    csv_file: containing your expression datasets
    control_columns : control_columns names
    replicate_columns_1 : replicate columns names for the first replicate
    replicate_columns_2 : replicate columns names for the second replicate
    replicate_columns_3 : replicate columns names for the third replicate
    fasta_file : fasta_file_for_the_assembly.
    Return: return_description
    """
    
    import pandas as pd
    import numpy as np
    initial_dataframe = pd.read_csv(csv_file, sep = "\t")
    sequence_file_train_read = list(filter(None,[x.strip() for x in open(fasta_file).readlines()]))
    sequence_train_dict = {}
    for i in sequence_file_train_read:
        if i.startswith(">"):
            genome_path = i.strip()
            if i not in sequence_train_dict:
                sequence_train_dict[i] = ""
                continue
        sequence_train_dict[genome_path] += i.strip()
    ids = list(map(lambda n: n.replace(">",""),sequence_train_dict.keys()))
    sequences = list(sequence_train_dict.values())
    sequence_dataframe = pd.DataFrame([(i,j) for i,j in zip(ids, sequences)])
    final_dataframe = pd.concat([initial_dataframe, sequence_dataframe])
    final_dataframe["length"] = final_dataframe["sequences"].apply(lambda n: len(n))
    hidden_layer_prep_dataframe = final_dataframe.dropna()
    length_nodes = np.array(hidden_layer_prep_dataframe["length"].to_list())
    control_nodes_weight = hidden_layer_prep_dataframe["control_columns"].\
                                                 apply(lambda n: sum(n)/len(n), axis = 1)
    first_replicate_weight = hidden_layer_prep_dataframe["replicate_columns_1"].\
                                                 apply(lambda n: sum(n)/len(n), axis = 1)
    second_replicate_weight = hidden_layer_prep_dataframe["replicate_columns_2"].\
                                                 apply(lambda n: sum(n)/len(n), axis = 1)
    third_replicate_weight = hidden_layer_prep_dataframe["replicate_columns_1"].\
                                                  apply(lambda n: sum(n)/len(n), axis = 1)
    control_node = {}
    control_node["node_control"] = np.array(control_nodes_weight.to_list())
    first_replicate_node = {}
    first_replicate_node["node_replicate1"] = np.array(first_replicate_weight.to_list())
    second_replicate_node = {}
    second_replicate_node["node_replicate2"] = np.array(second_replicate_weight.to_list())
    third_replicate_node = {}
    third_replicate_node["node_replicate3"] = np.array(third_replicate_weight.to_list())
    all_training_nodes = {**control_node,**first_replicate_node, \
                                            **second_replicate_node, **third_replicate_node}
    control_node_train = (length_nodes * all_training_nodes["control_node"]).sum()
    replicate_first_train = (length_nodes * all_training_nodes["first_replicate_node"]).sum()
    replicate_second_train = (length_nodes * all_training_nodes["second_replicate_node"]).sum()
    replicate_third_train = (length_nodes * all_training_nodes["third_replicate_node"]).sum()
    hidden_layer = np.array([control_node_train,replicate_first_train, replicate_second_train, replicate_third_train])
    return hidden_layer
