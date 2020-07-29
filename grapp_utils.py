import tensorflow as tf

def printTensors(pb_file):

    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for op in graph.get_operations():
        print(op.name)


path2pbfile= "Output/outColorNetOutputs_try11/k2tf_dir/output_graph.pb"
# path2pbfile = "Output/outColorNetOutputs_try11/frozen/frozen_cmodel.pb"
# path2pbfile = "Output/outColorNetOutputs_try11/model/model.pb"
printTensors(path2pbfile)