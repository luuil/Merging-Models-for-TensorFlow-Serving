# coding: utf-8
# author: luuil@outlook.com

r'''Export hyaudio model to TensorFlow Serving.'''

from saved_model import ServingInfo
from saved_model import export_serving_model
from saved_model import check_model_version_from_file


########### messages ###########

UPDATE_MESSAGE = 'add hyaudio_1 and hyaudio_2 model' # need be specified


########### export info ###########

@ServingInfo.export
def audio_hyaudio():
  signature_name       = 'hyaudio_1'
  frozen_pb            = 'audio/hyaudio.pb'
  input_signature_map  = { 'inputs':'hyaudio/vggish_input:0' }
  output_signature_map = { 'classes':'hyaudio/predict_classes:0',
                           'probs':'hyaudio/predict_probs:0'}
  return ServingInfo(frozen_pb, input_signature_map,
    output_signature_map, signature_name)

@ServingInfo.export
def audio_hyaudio():
  signature_name       = 'hyaudio_2'
  frozen_pb            = 'audio/hyaudio.pb'
  input_signature_map  = { 'inputs':'hyaudio/vggish_input:0' }
  output_signature_map = { 'classes':'hyaudio/predict_classes:0',
                           'probs':'hyaudio/predict_probs:0'}
  return ServingInfo(frozen_pb, input_signature_map,
    output_signature_map, signature_name)

# Will not export
# @ServingInfo.export
def audio_hyaudio():
  signature_name       = 'hyaudio_3'
  frozen_pb            = 'audio/hyaudio.pb'
  input_signature_map  = { 'inputs':'hyaudio/vggish_input:0' }
  output_signature_map = { 'classes':'hyaudio/predict_classes:0',
                           'probs':'hyaudio/predict_probs:0'}
  return ServingInfo(frozen_pb, input_signature_map,
    output_signature_map, signature_name)


if __name__ == '__main__':

  _EXPORT_DIR    = r'./serving'
  _FROZEN_DIR    = r'./frozen'
  _MODEL_VERSION = check_model_version_from_file(
    _EXPORT_DIR,
    update_message=UPDATE_MESSAGE,
    verbose=True)

  export_serving_model(_FROZEN_DIR, _EXPORT_DIR, _MODEL_VERSION,
    show_all_models=True, show_all_tensors=True)