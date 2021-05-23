import numpy as np
import time
from scipy.stats import entropy
import torch


########################################
# sampling utilities
########################################
def temperature(logits, temperature):
  try:
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    assert np.count_nonzero(np.isnan(probs)) == 0
  except:
    print ('overflow detected, use 128-bit')
    logits = logits.astype(np.float128)
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    probs = probs.astype(float)
  return probs

def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3] # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


########################################
# generation
########################################
def get_position_idx(event):
  return int(event.split('_')[-1])

def generate_fast(model, event2idx, idx2event, 
                  max_events=2048, max_bars=64, skip_check=False,
                  temp=1.2, top_p=0.9
                ):
  generated = [event2idx['Bar_None']]
  target_bars, generated_bars = max_bars, 0

  steps = 0
  time_st = time.time()
  cur_pos = 0
  failed_cnt = 0
  entropies = []
  while generated_bars < target_bars:
    dec_input = torch.tensor([generated]).long().to(next(model.parameters()).device)

    # sampling
    logits = model(
              dec_input, 
              keep_last_only=True, 
              attn_kwargs={'omit_feature_map_draw': len(generated) > 2}
            )
    logits = (logits[0]).cpu().detach().numpy()
    probs = temperature(logits, temp)
    word = nucleus(probs, top_p)
    word_event = idx2event[word]

    if not skip_check:
      if 'Beat' in word_event:
        event_pos = get_position_idx(word_event)
        if not event_pos >= cur_pos:
          failed_cnt += 1
          print ('[info] position not increasing, failed cnt:', failed_cnt)
          if failed_cnt >= 256:
            print ('[FATAL] model stuck, exiting with generated events ...')
            return generated
          continue
        else:
          cur_pos = event_pos
          failed_cnt = 0

    if 'Bar' in word_event:
      generated_bars += 1
      cur_pos = 0
      print ('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated)))
    if word_event == 'PAD_None' or (word_event == 'EOS_None' and generated_bars < target_bars - 3):
      continue
    elif word_event == 'EOS_None':
      print ('[info] gotten eos')
      generated.append(word)
      break

    generated.append(word)
    entropies.append(entropy(probs))
    steps += 1

    if len(generated) > max_events:
      print ('[info] max events reached')
      break

  print ('-- generated events:', len(generated))
  print ('-- time elapsed  : {:.2f} secs'.format(time.time() - time_st))
  print ('-- time per event: {:.2f} secs'.format((time.time() - time_st) / len(generated)))
  return generated[:-1], np.array(entropies)