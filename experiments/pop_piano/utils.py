import pickle
from models.music_performer_ape import MusicPerformer
from models.music_performer_spe import MusicPerformerSPE

def pickle_load(f):
  return pickle.load(open(f, 'rb'))

def pickle_dump(obj, f):
  pickle.dump(obj, open(f, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def load_model(model_conf, gpuid, vocab_size):
  if model_conf['pe_type'] == 'APE':
    model = MusicPerformer(
      vocab_size, model_conf['n_layer'], model_conf['n_head'], 
      model_conf['d_model'], model_conf['d_ff'], model_conf['d_embed'],
      favor_feature_dims=model_conf['feature_map']['n_dims']
    ).cuda(gpuid)
  elif model_conf['pe_type'] == 'SineSPE':
    model = MusicPerformerSPE(
      vocab_size, model_conf['n_layer'], model_conf['n_head'], 
      model_conf['d_model'], model_conf['d_ff'], model_conf['d_embed'],
      favor_feature_dims=model_conf['feature_map']['n_dims'],
      share_pe=model_conf['share_pe'], 
      share_spe_filter=model_conf['share_spe_filter'],
      spe_type='SineSPE',
      use_gated_filter=model_conf['use_gated_filter'],
      spe_module_params={
        'num_sines': model_conf['positional_encoder']['num_sines'],
        'num_realizations': model_conf['positional_encoder']['num_realizations']
      }
    ).cuda(gpuid)
  elif model_conf['pe_type'] == 'ConvSPE':
    model = MusicPerformerSPE(
      vocab_size, model_conf['n_layer'], model_conf['n_head'], 
      model_conf['d_model'], model_conf['d_ff'], model_conf['d_embed'],
      favor_feature_dims=model_conf['feature_map']['n_dims'],
      share_pe=model_conf['share_pe'], 
      share_spe_filter=model_conf['share_spe_filter'],
      spe_type='ConvSPE',
      use_gated_filter=model_conf['use_gated_filter'],
      spe_module_params={
        'kernel_size': model_conf['positional_encoder']['kernel_size'],
        'num_realizations': model_conf['positional_encoder']['num_realizations']
      }
    ).cuda(gpuid)

  return model