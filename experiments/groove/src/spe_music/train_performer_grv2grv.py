import sys, os, time
import argparse
import copy
from collections import defaultdict
import functools
from typing import Union, Optional
from pathlib import Path
import traceback

import bidict
from confugue import configurable, Configuration
import flatdict
import muspy
try:
  import neptune
except ImportError:
  traceback.print_exc()
  neptune = None
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import torch

from .model.music_performer import MusicPerformer
from .event_processor import BeatShiftTiming, EventRepresentationProcessor


DEVICE = 'cuda'


def log_epoch(log_file, log_data, is_init=False):
  if is_init:
    with open(log_file, 'w') as f:
      f.write('{:4} {:8} {:12} {:12}\n'.format('ep', 'steps', 'recons_loss', 'ep_time'))

  with open(log_file, 'a') as f:
    f.write('{:<4} {:<8} {:<12} {:<12}\n'.format(
      log_data['ep'], log_data['steps'], round(log_data['recons_loss'], 5), round(log_data['time'], 2)
    ))


@configurable
def train(
    model,
    ckpt_dir,
    train_dloader,
    val_dloaders=None,
    pretrained_param_path=None,
    optimizer_path=None,
    num_epochs=10000,
    warmup_steps=200,
    lr=1e-4,
    pad_index=None,
    feature_redraw_interval=1,
    log_interval=100,
    ckpt_interval=4,
    *, _cfg):

  train_steps = 0

  def train_epoch(epoch, model, train_dloader, optim, sched):
    model.train()
    recons_loss_rec = 0.
    accum_samples = 0

    print ('[epoch {:03d}] training ...'.format(epoch))
    print ('[epoch {:03d}] # batches = {}'.format(epoch, len(train_dloader)))
    st = time.time()

    for batch_idx, batch in enumerate(train_dloader):
      nonlocal train_steps
      train_steps += 1

      if neptune:
        neptune.log_metric('num_tokens', train_steps, batch.shape[1])

      if batch.shape[1] <= 2:
        print('Warning: length', batch.shape[1], 'at step', train_steps)
        continue

      batch = batch.to(DEVICE)
      batch_dec_inp = batch[:, :-1]
      batch_dec_tgt = batch[:, 1:]

      model.train()
      model.zero_grad()

      attn_kwargs = dict(
        omit_feature_map_draw=(train_steps - 1) % feature_redraw_interval != 0
      )
      if not attn_kwargs['omit_feature_map_draw']:
        print('Redrawing features')

      dec_logits = model(batch_dec_inp, attn_kwargs=attn_kwargs)
      losses = model.compute_loss(dec_logits, batch_dec_tgt, pad_index=pad_index)
      del dec_logits

      # clip gradient & update model
      losses['total_loss'].backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
      optim.step()

      # don't need gradients anymore
      losses = {k: v.detach() for k, v in losses.items()}

      recons_loss_rec += batch.size(0) * losses['recons_loss'].item()
      accum_samples += batch.size(0)

      # anneal learning rate
      if train_steps < warmup_steps:
        curr_lr = lr * train_steps / warmup_steps
        optim.param_groups[0]['lr'] = curr_lr
      else:
        sched.step()
        [curr_lr] = sched.get_last_lr()

      print (' -- epoch {:03d} | batch {:03d}: loss = {:.4f}, step = {}, time_elapsed = {:.2f} secs'.format(
        epoch, batch_idx, recons_loss_rec / accum_samples, train_steps, time.time() - st
      ))
      if neptune and train_steps % 10 == 0:
        neptune.log_metric('train_loss', train_steps, losses['total_loss'].item())
        neptune.log_metric('lr', train_steps, curr_lr)

      if not train_steps % log_interval:
        log_data = {
          'ep': epoch,
          'steps': train_steps,
          'recons_loss': recons_loss_rec / accum_samples,
        }

        model.eval()
        with torch.no_grad():
          for dset_name, dloader in val_dloaders.items() or []:
            vst = time.time()

            losses = []
            for batch in dloader:
              batch = batch.to(DEVICE)
              batch_dec_inp = batch[:, :-1]
              batch_dec_tgt = batch[:, 1:]
              dec_logits = model(batch_dec_inp)
              losses.append(model.compute_loss(dec_logits, batch_dec_tgt)['total_loss'].item())
            log_data[f'val_loss_{dset_name}'] = np.mean(losses)
            if neptune:
              neptune.log_metric(f'val_loss_{dset_name}', train_steps, log_data[f'val_loss_{dset_name}'])

            print(f'Validation on {dset_name} took {time.time() - vst:.2f} s')
            del vst

        log_data['time'] = time.time() - st

        log_epoch(
          os.path.join(ckpt_dir, 'log.txt'), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
        )

    print ('[epoch {:03d}] training completed\n  -- loss = {:.4f}\n  -- time elapsed = {:.2f} secs.'.format(
      epoch, recons_loss_rec / accum_samples, time.time() - st
    ))
    log_data = {
      'ep': epoch,
      'steps': train_steps,
      'recons_loss': recons_loss_rec / accum_samples,
      'time': time.time() - st
    }
    log_epoch(
      os.path.join(ckpt_dir, 'log.txt'), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
    )

    return recons_loss_rec / accum_samples

  if pretrained_param_path:
    pretrained_dict = torch.load(pretrained_param_path)
    pretrained_dict = {
      k:v for k, v in pretrained_dict.items() if 'feature_map.omega' not in k
    }
    model_state_dict = model.state_dict()
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)

  model.train()
  n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print ('# params:', n_params)

  opt_params = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = optim.Adam(opt_params, lr=lr)

  scheduler = _cfg['lr_scheduler'].configure(
    optim.lr_scheduler.CosineAnnealingWarmRestarts,
    optimizer=optimizer)
  if optimizer_path:
    optimizer.load_state_dict(
      torch.load(optimizer_path)
    )

  params_dir = os.path.join(ckpt_dir, 'params/')
  optimizer_dir = os.path.join(ckpt_dir, 'optim/')
  os.makedirs(ckpt_dir, exist_ok=True)
  os.makedirs(params_dir, exist_ok=True)
  os.makedirs(optimizer_dir, exist_ok=True)

  for ep in range(num_epochs):
    loss = train_epoch(epoch=ep+1, model=model, train_dloader=train_dloader,
                       optim=optimizer, sched=scheduler)

    if not (ep + 1) % ckpt_interval:
      torch.save(model.state_dict(),
        os.path.join(params_dir, 'ep{:03d}_loss{:.3f}_params.pt'.format(ep+1, loss))
      )
      torch.save(optimizer.state_dict(),
        os.path.join(optimizer_dir, 'ep{:03d}_loss{:.3f}_optim.pt'.format(ep+1, loss))
      )


class AugmentedDataset(muspy.Dataset):

  def __init__(self, dataset: muspy.Dataset,
               max_harm_tracks: int = None,
               harm_tracks_shuffle_prob: float = 0.4,
               track_drop_prob: float = 0.2,
               seed: int = 0):
    self.dataset = dataset
    self.max_harm_tracks = max_harm_tracks
    self.harm_tracks_shuffle_prob = harm_tracks_shuffle_prob
    self.track_drop_prob = track_drop_prob
    self.rng = np.random.default_rng(seed)

  def __getitem__(self, index) -> muspy.Music:
    return self._augment(self.dataset[index])

  def __len__(self) -> int:
    return len(self.dataset)

  def _augment(self, music: muspy.Music) -> muspy.Music:
    if [tr.name for tr in music.tracks] != [
        'BB Bass', 'BB Drums', 'BB Guitar', 'BB Piano', 'BB Strings']:
      print('Warning: unexpected track names', *[tr.name for tr in music.tracks])
      print('Music:', music)

    harm_track_ids = [2, 3, 4]  # Guitar, Piano, Strinfs

    # Shuffle Guitar, Piano, Synth tracks randomly
    if self.rng.random() < self.harm_tracks_shuffle_prob:
      self.rng.shuffle(harm_track_ids)

    # Keep only the specified number of harmonic tracks; prefer non-empty ones
    if self.max_harm_tracks is not None:
      harm_track_ids.sort(key=lambda i: 0 if music.tracks[i].notes else 1)
      harm_track_ids = harm_track_ids[:self.max_harm_tracks]

    music.tracks[2:] = [music.tracks[i] for i in harm_track_ids]

    # Cut off up to 64 bars at the beginning (4 at a time)
    total_bars = music.get_end_time() // music.resolution // 4
    if self.rng.random() < 0.25 and total_bars >= 8:
      offset = music.resolution * 4 * 4 * int(self.rng.integers(1, min(64, total_bars // 2) // 4, endpoint=True))
      music.adjust_time(lambda t: t - offset)
      # Remove notes manually because remove_invalid() is buggy
      for track in music:
        track.notes = [n for n in track.notes if n.end >= 0]

    # Drop tracks at random, but keep at least one non-empty
    nonempty_track_ids = [i for i, tr in enumerate(music.tracks) if tr.notes]
    self.rng.shuffle(nonempty_track_ids)
    for i in nonempty_track_ids[1:]:
      if self.rng.random() < self.track_drop_prob:
        music.tracks[i].notes.clear()

    # Transpose randomly
    music.transpose(self.rng.integers(-5, 6, endpoint=True))
    # Fix invalid notes after transposition
    for track in music.tracks:
      for note in track.notes:
        if note.pitch < 0:
          note.pitch += 12
        elif note.pitch > 127:
          note.pitch -= 12

    assert len(music.tracks) == 2 + self.max_harm_tracks

    return music


def collate_padded(batch, pad_value=0, max_len=np.inf):
  max_len = min(max_len, max(len(x) for x in batch))
  batch = [
    np.pad(x, [(0, max(0, max_len - len(x)))], constant_values=pad_value)[:max_len]
    for x in batch]
  return torch.as_tensor(batch)


@configurable
def make_representation(*, _cfg):
  representation = _cfg['representation'].configure(
    EventRepresentationProcessor,
    timing=_cfg['timing'].configure(
      BeatShiftTiming
    ),
    use_end_of_sequence_event=True)
  vocab = dict(representation.vocab)
  start_id = max(vocab.values()) + 1
  vocab[('bos',)] = start_id
  end_id = vocab[('eos',)]
  representation.vocab = bidict.frozenbidict(vocab)

  return representation, start_id, end_id


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-dir', type=str, required=True)
  parser.add_argument('--name', type=str)
  parser.add_argument('--load-params', type=str)
  parser.add_argument('--load-optim', type=str)
  args = parser.parse_args()

  cfg_path = os.path.join(args.model_dir, 'config.yaml')
  cfg = Configuration.from_yaml_file(cfg_path)

  global neptune
  if neptune:
    try:
      neptune.init()
      neptune.create_experiment(
        args.name or args.model_dir,
        upload_source_files=[],
        params=dict(flatdict.FlatterDict(cfg.get(), delimiter='.')))
    except neptune.exceptions.NeptuneException:
      neptune = None
      traceback.print_exc()

  seed = cfg.get('seed', 0)
  np.random.seed(seed)
  torch.random.manual_seed(seed)

  representation, start_id, end_id = cfg.configure(make_representation)
  print('Vocab size:', len(representation.vocab))

  def encode(music: muspy.Music):
    encoded = representation.encode(music)
    encoded = np.concatenate([[start_id], encoded])
    return encoded

  data_train = muspy.MusicDataset(cfg.get('train_data_path'))
  data_train = cfg['data_augmentation'].configure(
    AugmentedDataset, dataset=data_train, seed=seed)
  data_train_pt = data_train.to_pytorch_dataset(factory=encode)

  model = cfg['model'].configure(
    MusicPerformer,
    n_token=len(representation.vocab),
  ).to(DEVICE)

  train_loader = cfg['data_loader'].configure(
    DataLoader,
    dataset=data_train_pt,
    collate_fn=functools.partial(
      collate_padded, pad_value=end_id, max_len=model.max_len),
    batch_size=1,
    shuffle=True,
    num_workers=24)

  val_loaders = {}
  if cfg['val_data_paths']:
    val_loaders = {
      name: cfg['val_data_loader'].configure(
        DataLoader,
        dataset=muspy.MusicDataset(path).to_pytorch_dataset(factory=encode),
        collate_fn=functools.partial(
          collate_padded, pad_value=end_id, max_len=model.max_len),
        batch_size=1,
        shuffle=False,
        num_workers=24
      )
      for name, path in cfg.get('val_data_paths').items()
    }

  cfg['training'].configure(train,
    model=model,
    ckpt_dir=args.model_dir,
    pretrained_param_path=args.load_params,
    optimizer_path=args.load_optim,
    train_dloader=train_loader,
    val_dloaders=val_loaders,
    pad_index=end_id)

  cfg.get_unused_keys(warn=True)


if __name__ == '__main__':
  main()
