import torch

'''
This file contains implementation of PE evaluation metrics
("translation invariance" and "monotonicity)
proposed in Wang et al. (ICLR 2021)
'''

################################################
# helper functions for metric "monotonicity"
################################################
def inversion_counter(arr, n): 
    temp_arr = [0]*n 
    return _merge_sort(arr, temp_arr, 0, n-1) 
  
# This Function will use MergeSort to count inversions 
  
def _merge_sort(arr, temp_arr, left, right): 
    inv_count = 0
    if left < right: 
        mid = (left + right)//2
        inv_count += _merge_sort(arr, temp_arr,  
                                    left, mid) 
  
        inv_count += _merge_sort(arr, temp_arr,  
                                  mid + 1, right) 
  
        inv_count += merge(arr, temp_arr, left, mid, right) 
    return inv_count 
  

def merge(arr, temp_arr, left, mid, right): 
    i = left     
    j = mid + 1 
    k = left     
    inv_count = 0

    while i <= mid and j <= right: 
        if arr[i] <= arr[j]: 
            temp_arr[k] = arr[i] 
            k += 1
            i += 1
        else: 
            temp_arr[k] = arr[j] 
            inv_count += (mid-i + 1) 
            k += 1
            j += 1
  
    while i <= mid: 
        temp_arr[k] = arr[i] 
        k += 1
        i += 1
  
    while j <= right: 
        temp_arr[k] = arr[j] 
        k += 1
        j += 1
  
    for loop_var in range(left, right + 1): 
        arr[loop_var] = temp_arr[loop_var] 
          
    return inv_count 

################################################
# metrics below support causal attention only
################################################
def calc_transition_invariance(attn_map, min_pos=0, max_pos=None, max_offset=None):
  '''
  attn_map: tensor with shape (query_length, key_length)
  '''
  if max_pos is None or max_pos > attn_map.size(0):
    max_pos = attn_map.size(0)
  if max_offset is None:
    max_offset = max_pos - 1
  
  attn_map = attn_map[min_pos:max_pos, :max_pos]

  diag_start = min_pos
  diag_end = diag_start - max_offset - 1

  _sum, _sum_squared = 0., 0.
  size_sum = 0
  invar_weighted_sum = 0.
  for offset in range(diag_start, diag_end, -1):
    tau = torch.diagonal(attn_map, offset)
    _sum += tau.sum().item()
    _sum_squared += (tau ** 2).sum().item()
    size_sum += tau.size(0)
    invar_weighted_sum += torch.var(tau, unbiased=False) * tau.size(0)

  total_var = (_sum_squared / size_sum) - (_sum / size_sum) ** 2
  avg_invar = invar_weighted_sum / size_sum

  return (avg_invar / total_var).item()

def calc_monotonicity(attn_map, min_pos=0, max_pos=None, max_offset=None):
  '''
  attn_map: tensor with shape (query_length, key_length)
  '''
  if max_pos is None or max_pos > attn_map.size(0):
    max_pos = attn_map.size(0)
  if max_offset is None:
    max_offset = max_pos - 1
  
  attn_map = attn_map[:max_pos, :max_pos]

  pairs_cnt = 0
  violated_pairs_cnt = 0
  for q_pos in range(min_pos, max_pos):
    attn_row_q = attn_map[q_pos][:q_pos + 1]
    if attn_row_q.size(0) == 1:
      continue
    elif attn_row_q.size(0) > max_offset + 1:
      attn_row_q = attn_row_q[-max_offset-1:]

    len_row_q = attn_row_q.size(0)
    pairs_cnt_at_pos = (len_row_q * (len_row_q - 1)) / 2
    
    invs = inversion_counter(list(attn_row_q), len_row_q)
    pairs_cnt += pairs_cnt_at_pos
    violated_pairs_cnt += invs
    if not q_pos % 512:
      print (q_pos, invs, pairs_cnt_at_pos)

  return violated_pairs_cnt / pairs_cnt 
