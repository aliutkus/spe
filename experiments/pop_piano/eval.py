import sys, random
sys.path.append('./models')

from dataloader import REMIFullSongTransformerDataset
from torch.utils.data import DataLoader

from utils import pickle_load, load_model
import torch
import numpy as np

import yaml
train_conf_path = sys.argv[1]
train_conf = yaml.load(open(train_conf_path, 'r'), Loader=yaml.FullLoader)

ckpt_path = sys.argv[2]
split = './pickles/test_pieces.pkl'
segment_width = 128
max_eval_len = 4096
trained_len = train_conf['model']['max_len']

train_conf_ = train_conf['training']
redraw_prob = train_conf_['feat_redraw_prob']
gpuid = train_conf_['gpuid']
torch.cuda.set_device(gpuid)

def eval(model, dloader):
  model.eval()
  loss_rec = np.zeros((max_eval_len,))
  counts = np.zeros((max_eval_len,))

  avg_losses = []
  avg_ext_losses = []
  print ('>> per sample NLLs')
  with torch.no_grad():
    for batch_idx, batch_samples in enumerate(dloader):
      batch_dec_inp = batch_samples['dec_input'].cuda(gpuid)
      batch_dec_tgt = batch_samples['dec_target'].cuda(gpuid)
      batch_inp_lens = batch_samples['length']

      omit_feature_map_draw = random.random() > redraw_prob
      dec_logits = model(
                      batch_dec_inp, 
                      attn_kwargs={'omit_feature_map_draw': omit_feature_map_draw}
                    )
      losses = model.compute_loss(dec_logits, batch_dec_tgt, reduction='none')['recons_loss']
      losses = losses.cpu().detach().numpy()

      loss_rec[:batch_inp_lens[0] - 1] += losses[:batch_inp_lens[0] - 1]
      counts[:batch_inp_lens[0] - 1] += 1
      # print (counts[:10], counts[-10:])

      if batch_inp_lens <= trained_len:
        avg_losses.append( np.mean(losses[:batch_inp_lens[0] - 1]) )
        print ('  - samp #{:03d}: within = {:.3f}'.format(
          batch_idx + 1,
          np.mean(losses[:batch_inp_lens[0] - 1])
        ))
      else:
        end_t = min(max_eval_len, batch_inp_lens[0] - 1)
        avg_losses.append( np.mean(losses[:trained_len]) )
        avg_ext_losses.append( np.mean(losses[trained_len:end_t]))
        print ('  - samp #{:03d}: within = {:.3f} | extrap. = {:.3f}'.format(
          batch_idx + 1,
          np.mean(losses[:trained_len]), 
          np.mean(losses[trained_len:end_t])
        ))

  # remove positions w/o any samples before calculating avg
  zero_counts_idx = np.where(counts == 0)[0]
  if len(zero_counts_idx) > 0:
    counts = counts[ : zero_counts_idx[0] ]
    loss_rec = loss_rec[ : zero_counts_idx[0] ]

  return loss_rec / counts, avg_losses, avg_ext_losses

if __name__ == '__main__':
  test_dset = REMIFullSongTransformerDataset(
    './remi_dataset', './pickles/remi_vocab.pkl', 
    do_augment=False,
    model_dec_seqlen=max_eval_len + 1, 
    model_max_bars=512,
    pieces=pickle_load(split)
  )
  test_dloader = DataLoader(test_dset, batch_size=1, shuffle=True, num_workers=8)

  model = load_model(train_conf['model'], gpuid, test_dset.vocab_size)
  pretrained_dict = torch.load(ckpt_path)
  pretrained_dict = {
    k:v for k, v in pretrained_dict.items() if 'feature_map.omega' not in k
  }
  model_state_dict = model.state_dict()
  model_state_dict.update(pretrained_dict)
  model.load_state_dict(model_state_dict)

  losses, avg_loss, avg_ext_loss = eval(model, test_dloader)

  print ('\n>> evaluated model {} {}'.format(
    type(model), 
    model.spe_type if hasattr(model, 'spe_type') else ''
  ))
  trained_has_end = False
  for seg_start in range(0, len(losses), segment_width):
    if seg_start >= trained_len and not trained_has_end:
      print ('^^^^^^^^^ end of trained seqlen ^^^^^^^^^^^')
      trained_has_end = True

    seg_end = min(len(losses), seg_start + segment_width)
    print ('[pos {:6} -- {:6}] nll_loss = {:.3f}'.format(
      str(seg_start), str(seg_end), np.mean(losses[ seg_start : seg_end ])
    ))