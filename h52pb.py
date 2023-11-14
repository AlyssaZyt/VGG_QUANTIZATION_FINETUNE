from tensorflow.compat.v1.graph_util import convert_variables_to_constants
from tensorflow.keras.models import load_model
import tensorflow.compat.v1 as tf
import tensorflow_model_optimization as tfmot

def h5_2_pb(keras_path):
    pb_path = keras_path.replace('.h5', '.pb')
    tf.disable_eager_execution() #disable eager mode
    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(0)
    with tfmot.quantization.keras.quantize_scope():
        keras_model = load_model(keras_path)
    session = tf.keras.backend.get_session()
    input_names = [input.op.name for input in keras_model.inputs]
    output_names = [out.op.name for out in keras_model.outputs]
    frozen_gdef = convert_variables_to_constants(session,
                                                 session.graph_def,
                                                 output_names)
    with tf.io.gfile.GFile(pb_path, mode = 'wb') as f:
        f.write(frozen_gdef.SerializeToString())
    print(12*'=', 'Save pb model to: ', pb_path, 12*'='+'\n')

keras_path = 'path to h5 model'
h5_2_pb((keras_path))