# coding: utf-8
# author: luuil@outlook.com

r"""Universal serialization format for TensorFlow models."""

import os
import tensorflow as tf
from os.path import join as pjoin
from abc import ABC


_MODELS_TO_EXPORT     = [] # global model info list
_VERSION_FILE         = 'current_version'
_EXPORTED_MODELS_FILE = 'exported_models'

tf.logging.set_verbosity(tf.logging.INFO)

########################## FrozenModel ##########################

class FrozenModel(object):
  """Prepare frozen model(.pb) for exporting to tf-serving."""
  def __init__(self, sess, frozen_pb):
    super(FrozenModel, self).__init__()
    assert tf.gfile.Exists(frozen_pb), \
      'not exists: {}'.format(frozen_pb)
    
    self._sess = sess
    self._frozen_pb = frozen_pb

  def load_graph(self, input_map=None, return_elements=None, name=""):
    """Load graph def from frozen model file(.pb) to sepecfic graph.

    Args:
      input_map: A dictionary mapping input names (as strings) in graph_def
        to Tensor objects. The values of the named input tensors in the 
        imported graph will be re-mapped to the respective Tensor values.
      return_elements: A list of strings containing operation names in 
        graph_def that will be returned as Operation objects; and/or tensor
        names in graph_def that will be returned as Tensor objects.
      name: STRING. Name scope for loaded graph.
        e.g. Give a tensor name `a/b:0`, and name=`import`. then in
        the new graph, the name will be `import/a/b:0`.

    Returns:
      List of `Operation` and/or `Tensor` objects from the imported graph,
      corresponding to the names in `return_elements`.
    """
    self._graph_name_prefix = (name + '/') if len(name) > 0 else ""
    with self._sess.graph.as_default(),\
    tf.gfile.FastGFile(self._frozen_pb, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      
      # `ops_and_tensors` is a list of `Operation` and/or `Tensor` objects
      # from the imported graph, corresponding to the names in `return_elements`.
      ops_and_tensors = tf.import_graph_def(
        graph_def,
        input_map=input_map,
        return_elements=return_elements,
        name=name
      )
      return ops_and_tensors
  
  def build_signature_def(self, input_signature_map, output_signature_map,
    signature_name):
    """Build a SignatureDef protocol buffer.
    
    Args:
      input_signature_map: Inputs of the SignatureDef defined as a proto
        map of string to tensor name.
      output_signature_map: Outputs of the SignatureDef defined as a proto
        map of string to tensor name.
      signature_name: Name of SignatureDef protocol buffer.

    Returns:
      A SignatureDef protocol buffer constructed based on the supplied arguments.
    """
    def tensor_info(tensor_name):
      tensor = self._sess.graph.get_tensor_by_name(
        self._graph_name_prefix + tensor_name)
      info = tf.saved_model.utils.build_tensor_info(tensor)
      return info
    
    inputs = { tensor_key:tensor_info(tensor_name) \
      for (tensor_key, tensor_name) in input_signature_map.items() }
    outputs = { tensor_key:tensor_info(tensor_name) \
      for (tensor_key, tensor_name) in output_signature_map.items() }
    
    signature_def = (
      tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
      )
    )
    return {signature_name: signature_def}


def frozen_model_wrapper(sess, frozen_pb,
  input_signature_map, output_signature_map, signature_name,
  input_map=None, return_elements=None):
  hyaudio = FrozenModel(sess, frozen_pb)
  _useless_ops_and_tensors = hyaudio.load_graph(input_map, return_elements,
    name=signature_name)
  signature_def_map = hyaudio.build_signature_def(input_signature_map,
    output_signature_map, signature_name)
  return signature_def_map


########################## FrozenToServing ##########################

class FrozenToServing(object):
  """Export frozen model signature_def to serving."""
  def __init__(self, export_dir, model_version):
    super(FrozenToServing, self).__init__()
    export_path = pjoin(
      tf.compat.as_bytes(export_dir),
      tf.compat.as_bytes(str(model_version)))
    self._builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    self._signature_def_map = dict()

  def add_frozen_model(self, signature_def_map):
    self._signature_def_map.update(signature_def_map)

  def export(self, sess):
    with sess.graph.as_default():
      legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
      self._builder.add_meta_graph_and_variables(
          sess,
          [tf.saved_model.tag_constants.SERVING],
          signature_def_map = self._signature_def_map,
          legacy_init_op=legacy_init_op
      )
      self._builder.save()

  def __str__(self):
    return str(self._signature_def_map)

########################## ServingExporter ##########################

class ServingInfo(object):
  """docstring for ServingInfo"""
  def __init__(self, frozen_pb,
    input_signature_map, output_signature_map, signature_name,
    input_map=None, return_elements=None):
    super(ServingInfo, self).__init__()
    self.frozen_pb = frozen_pb
    self.input_signature_map = input_signature_map
    self.output_signature_map = output_signature_map
    self.signature_name = signature_name
    self.input_map = input_map
    self.return_elements = return_elements

  @staticmethod
  def export(serving_info_func):
    _MODELS_TO_EXPORT.append(serving_info_func())

  def __str__(self):
    string = 'ServingInfo:\n'\
      '\tsignature_name: "{}"\n'\
      '\tfrozen_pb: "{}"\n'\
      '\tinput_signature_map: {}\n'\
      '\toutput_signature_map: {}\n'\
      '\tinput_map: {}\n'\
      '\treturn_elements: {}\n'.format(
        self.signature_name,
        self.frozen_pb,
        self.input_signature_map ,
        self.output_signature_map,
        self.input_map,
        self.return_elements)
    return string


class LimitedSessBase(ABC):
  """Source Limited Session"""
  def __init__(self):
    super(LimitedSessBase, self).__init__()
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    self._sess = tf.Session(config=config, graph=tf.Graph())

  def close(self):
    self._sess.close()


class ServingExporter(LimitedSessBase):
  """Exporter for export serving model."""
  def __init__(self, export_dir, model_version):
    super(ServingExporter, self).__init__()
    self._fts = FrozenToServing(export_dir, model_version)

  def add_model(self, serving_info):
    signature_def_map = frozen_model_wrapper(
      self._sess,
      serving_info.frozen_pb,
      serving_info.input_signature_map,
      serving_info.output_signature_map,
      serving_info.signature_name,
      serving_info.input_map,
      serving_info.return_elements)
    self._fts.add_frozen_model(signature_def_map)

  def add_models(self, serving_infos):
    for serving_info in serving_infos:
      self.add_model(serving_info)
    tf.logging.info('Signature def map:\n' + str(self._fts))

  def show_all_tensors(self, scope):
    with self._sess.graph.as_default():
      tensor_names = [n.name for n in self._sess.graph.as_graph_def().node
        if n.name.startswith(scope)]
    tensor_names_str = '\n\t'.join(tensor_names)
    tf.logging.info('Tensor scope: {}\n\t{}\n'.format(scope,
      tensor_names_str))

  def export(self):
    self._fts.export(self._sess)
    super(ServingExporter, self).close()


def export_serving_model(frozen_dir, export_dir, model_version,
  show_all_models=False, show_all_tensors=False):
  se = ServingExporter(export_dir, model_version)
  _MODELS_TO_EXPORT2 = []
  for m in _MODELS_TO_EXPORT:
    m.frozen_pb = pjoin(frozen_dir, m.frozen_pb)
    _MODELS_TO_EXPORT2.append(m)
  se.add_models(_MODELS_TO_EXPORT2)
  all_models = [m.signature_name for m in _MODELS_TO_EXPORT2]
  if show_all_tensors:
    for scope in all_models:
      se.show_all_tensors(scope)
  if show_all_models:
    for m in _MODELS_TO_EXPORT2:
      tf.logging.info(m)
    tf.logging.info('Models count: {}'.format(len(_MODELS_TO_EXPORT2)))
  se.export()

########################## Others ##########################

def freezing_model(model_dir, output_node_names, output_dir=None, name=None):
  """Extract the sub graph defined by the output nodes and convert 
  all its variables into constant.
  
  Args:
    model_dir: Root folder containing the checkpoint state file.
    output_node_names: List of string of output node names.

  Returns:
    Path to frozen model.
  """
  if not tf.gfile.Exists(model_dir):
    raise AssertionError(
      "Export directory doesn't exists. Please specify an export "
      "directory: %s" % model_dir)

  if not output_node_names:
    tf.logging.error("You need to supply the name of a node.")
    return -1

  # We retrieve our checkpoint fullpath
  checkpoint = tf.train.get_checkpoint_state(model_dir)
  input_checkpoint = checkpoint.model_checkpoint_path
  tf.logging.info('Input checkpoint is: {}'.format(input_checkpoint))
  
  # We precise the file fullname of our freezed graph
  output_path = join((output_dir if output_dir is not None else model_dir),
      "frozen_{}.pb".format((name if name is not None else "model"))
    )

  # We clear devices to allow TensorFlow to control on which device it will load operations
  clear_devices = True

  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True

  # We start a session using a temporary fresh Graph
  with tf.Session(graph=tf.Graph(), config=sess_config) as sess:
    # We import the meta graph in the current default Graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We restore the weights
    saver.restore(sess, input_checkpoint)

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, # The session is used to retrieve the weights
      sess.graph.as_graph_def(), # The graph_def is used to retrieve the nodes 
      output_node_names=output_node_names # The output node names are used to select the usefull nodes
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_path, "wb") as f:
      f.write(output_graph_def.SerializeToString())
    tf.logging.info("%d ops in the final graph." % len(output_graph_def.node))

  return output_path

def check_model_version_from_dir(export_dir, verbose=False):
  assert os.path.exists(export_dir), "Not exists: {}".format(
    export_dir)
  versions = [int(d) for d in os.listdir(export_dir)
    if os.path.isdir(pjoin(export_dir,d))]
  cur_ver = max(versions) if len(versions) > 0 else 0
  if verbose:
    tf.logging.info('Previous model versions: {}'.format(versions))
    tf.logging.info('Current model version: {}'.format(cur_ver))
  return (cur_ver + 1)

def check_model_version_from_file(export_dir,
  version_filename=_VERSION_FILE,
  exported_models_filename=_EXPORTED_MODELS_FILE,
  update_message=None,
  verbose=False):
  def _prepend(fhandle, content):
    fhandle.seek(0)
    fhandle.write(content)
    fhandle.truncate() # 清空后面的部分

  def _check_exported_models():
    signature_name_list = [serving_info.signature_name for serving_info in _MODELS_TO_EXPORT]
    signature_name_list_str = '\n'.join(signature_name_list)
    exported_models_file = pjoin(export_dir, exported_models_filename)
    if not os.path.exists(exported_models_file):
      with open(exported_models_file, 'w', encoding='utf-8') as ef:
        ef.write(signature_name_list_str)
        return ','.join(signature_name_list)
    else:
      with open(exported_models_file, 'r+', encoding='utf-8') as ef:
        old_signature_name_list = ef.read().split('\n')
        newly_signature_name_list = [item for item in signature_name_list \
          if item not in old_signature_name_list]
        if len(newly_signature_name_list) > 0:
          _prepend(ef, signature_name_list_str)
        return ','.join(newly_signature_name_list)

  assert os.path.exists(export_dir), "Not exists: {}".format(
    export_dir)
  cur_ver = 0
  version_file = pjoin(export_dir, version_filename)
  newly_models = _check_exported_models()
  if not os.path.exists(version_file):
    with open(version_file, 'w', encoding='utf-8') as vf:
      new_line = '{version}\t{newly_models}\t"{msg}"\n'.format(version=(cur_ver+1),
        newly_models=newly_models, msg=update_message)
      vf.write(new_line)
  else:
    with open(version_file, 'r+', encoding='utf-8') as vf:
      old_content = vf.read()
      vf.seek(0)
      first_line_contents = vf.readline().strip('\n').split('\t')
      cur_ver = int(first_line_contents[0])
      new_line = '{version}\t{newly_models}\t"{msg}"\n'.format(version=(cur_ver+1),
        newly_models=newly_models, msg=update_message)
      _prepend(vf, new_line+old_content)
  if verbose:
    tf.logging.info('Current model version: {}'.format(cur_ver))
  return (cur_ver + 1)


########################## Tests ##########################


def test_hyaudio():
  

  @ServingInfo.export
  def hyaudio_serving_info():
    model_dir = r'D:\hyml-audio\singing-or-not\data\frozen'
    model_name = 'frozen_hyaudio.pb'
    frozen_pb = pjoin(model_dir, model_name)
    input_signature_map = { 'inputs': 'hyaudio/vggish_input:0' }
    output_signature_map = { 'classes': 'hyaudio/predict_classes:0',
                             'probs': 'hyaudio/predict_probs:0'}
    signature_name = "hyaudio"
    
    return ServingInfo(frozen_pb, input_signature_map,
      output_signature_map, signature_name)
  
  @ServingInfo.export
  def vggish_serving_info():
    model_dir = r'D:\hyml-audio\singing-or-not\data\frozen'
    model_name = 'frozen_vggish.pb'
    frozen_pb = pjoin(model_dir, model_name)
    input_signature_map = { 'inputs': 'vggish/input_features:0' }
    output_signature_map = { 'outputs': 'vggish/embedding:0' }
    signature_name = "vggish"
    
    return ServingInfo(frozen_pb, input_signature_map,
      output_signature_map, signature_name)

  print(_MODELS_TO_EXPORT)

  EXPORT_DIR = "./serving"
  FROZEN_DIR = "./frozen"
  MODEL_VERSION = check_model_version(EXPORT_DIR)
  export_serving_model(FROZEN_DIR, EXPORT_DIR, MODEL_VERSION, show_all_tensors=True)

if __name__ == '__main__':
  
  # test_hyaudio()
  print(check_model_version('./serving', verbose=True))

  pass