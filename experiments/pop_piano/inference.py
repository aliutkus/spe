import sys, os
sys.path.append('./models')
import torch
import numpy as np

from utils import pickle_load, load_model
from convert2midi import event_to_midi
from generate_utils import generate_fast

import yaml
train_conf_path = sys.argv[1]
inf_conf_path = sys.argv[2]
train_conf = yaml.load(open(train_conf_path, 'r'), Loader=yaml.FullLoader)
inf_conf = yaml.load(open(inf_conf_path, 'r'), Loader=yaml.FullLoader)

REMI_MODEL_VOCAB_SIZE = 333
gpuid = inf_conf['gpuid']
torch.cuda.set_device(gpuid)

def word2event(word_seq, idx2event):
  return [ idx2event[w] for w in word_seq ]

if __name__ == "__main__":
  event2idx, idx2event = pickle_load('./pickles/remi_vocab.pkl')

  ckpt_path = inf_conf['ckpt_path']
  n_pieces = inf_conf['gen_n_pieces']
  gen_output_dir = inf_conf['gen_output_dir']
  gen_max_events = inf_conf['gen_max_events']
  gen_max_bars = inf_conf['gen_max_bars']
  sampling_temp = inf_conf['sampling']['temp']
  sampling_top_p = inf_conf['sampling']['top_p']

  model = load_model(train_conf['model'], gpuid, REMI_MODEL_VOCAB_SIZE)

  pretrained_dict = torch.load(ckpt_path)
  pretrained_dict = {
    k:v for k, v in pretrained_dict.items() if 'feature_map.omega' not in k
  }
  model_state_dict = model.state_dict()
  model_state_dict.update(pretrained_dict)
  model.load_state_dict(model_state_dict)
  model.eval()
  print ('[info] trained model weights loaded')

  if not os.path.exists(gen_output_dir):
    os.makedirs(gen_output_dir)
  
  all_ents = np.zeros((1, gen_max_events))
  with torch.no_grad():
    for p in range(n_pieces):
      print ('model:', type(model), '{}'.format(model.spe_type if hasattr(model, 'spe_type') else ''))
      print ('piece:', p+1)
      out_file = os.path.join(gen_output_dir, 
        'samp{:02d}'.format(p + 1)
      )
    
      song, entropies = generate_fast(model, event2idx, idx2event, 
        max_bars=gen_max_bars, max_events=gen_max_events, skip_check=False,
        temp=sampling_temp, top_p=sampling_top_p
      )

      song = word2event(song, idx2event)
      print (*song, sep='\n', file=open(out_file + '.txt', 'w'))
      event_to_midi(song, out_file + '.mid')

      # [optional] Save per-token entropies during generation
      # print (entropies.shape)
      # all_ents = np.concatenate(
      #   (all_ents, np.expand_dims(entropies, axis=0)),
      #   axis=0
      # )
      # print (all_ents.shape)
      # np.save(os.path.join(gen_output_dir, 'entropies'), all_ents[1:, :])