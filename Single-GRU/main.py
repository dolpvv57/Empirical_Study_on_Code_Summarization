from torch.utils.data import TensorDataset, DataLoader
from vocabulary import Vocabulary
from train import train
from test import test
from datetime import datetime
from dataset_obj import DatasetObject2

import config
import logging
import os
import torch
import itertools
import random
import time

def build_vcblry(dataset, vtype):
  vcblry = Vocabulary(vtype)
  for seq in dataset:
    vcblry.add_sequence(seq)
  vcblry.trim(config.max_vcblry_size)
  if not os.path.exists(config.vcblry_base_path):
    os.makedirs(config.vcblry_base_path)
  vcblry.save(config.vcblry_base_path)
  return vcblry

def to_idx_seq(batch, vcblry):
  idx_seqs = []
  for word_seq in batch:
    cur_idx_seq = []
    for word in word_seq:
      if word not in vcblry.word2idx:
        cur_idx_seq.append(vcblry.word2idx['<UNK>'])
      else:
        cur_idx_seq.append(vcblry.word2idx[word])
    cur_idx_seq.append(vcblry.word2idx['<EOS>'])
    idx_seqs.append(cur_idx_seq)
  return idx_seqs

def get_seq_lens(batch):
  seq_lens = []
  for seq in batch:
    seq_lens.append(len(seq))
  return seq_lens

def padding(batch, vcblry):
  padded_batch = list(itertools.zip_longest(*batch, fillvalue=vcblry.word2idx['<PAD>']))
  padded_batch = [list(x) for x in padded_batch]
  return torch.tensor(padded_batch, device=config.device).long()

def my_collate_fn(batch, keycode_vcblry, sbt_vcblry, nl_vcblry, is_raw_nl=False):
  keycode_batch = []
  sbt_batch = []
  nl_batch = []
  for entry in batch[0]:
    keycode_batch.append(entry[0])
    sbt_batch.append(entry[1])
    nl_batch.append(entry[2])

  keycode_batch = to_idx_seq(keycode_batch, keycode_vcblry)
  sbt_batch = to_idx_seq(sbt_batch, sbt_vcblry)
  if not is_raw_nl:
    nl_batch = to_idx_seq(nl_batch, nl_vcblry)

  # 先get seq lengths再padding
  keycode_seq_lens = get_seq_lens(keycode_batch)
  sbt_seq_lens = get_seq_lens(sbt_batch)
  nl_seq_lens = get_seq_lens(nl_batch)

  keycode_batch = padding(keycode_batch, keycode_vcblry)
  sbt_batch = padding(sbt_batch, sbt_vcblry)
  if not is_raw_nl:
    nl_batch = padding(nl_batch, nl_vcblry)

  return keycode_batch, keycode_seq_lens, sbt_batch, sbt_seq_lens, nl_batch, nl_seq_lens

def my_collate_fn2(batch, keycode_vcblry, nl_vcblry, is_raw_nl=False):
  keycode_batch = []
  nl_batch = []
  for entry in batch[0]:
    keycode_batch.append(entry[0])
    nl_batch.append(entry[1])

  keycode_batch = to_idx_seq(keycode_batch, keycode_vcblry)
  if not is_raw_nl:
    nl_batch = to_idx_seq(nl_batch, nl_vcblry)

  # 先get seq lengths再padding
  keycode_seq_lens = get_seq_lens(keycode_batch)
  nl_seq_lens = get_seq_lens(nl_batch)

  keycode_batch = padding(keycode_batch, keycode_vcblry)
  if not is_raw_nl:
    nl_batch = padding(nl_batch, nl_vcblry)

  return keycode_batch, keycode_seq_lens, nl_batch, nl_seq_lens

def set_random_seed(seed):
  random.seed(seed)
  torch.manual_seed(seed) # 为CPU设置随机种子
  torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
  torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子

def get_logger(path):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

if __name__ == '__main__':

  # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
  
  set_random_seed(config.rand_seed)
  logger = get_logger(config.dataset_base_path + 
                      'log/{}_{}'.format(config.log_name, time.strftime("%Y%m%d")))

  # 训练集准备
  logger.info('Init trainset...')
  trainset = DatasetObject2(config.keycode_trainset_path, config.nl_trainset_path)
  train_loader = DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True,
      collate_fn=lambda *x: my_collate_fn2(x, keycode_vcblry, nl_vcblry))  
  logger.info('Done')

  # 验证集、测试集准备
  logger.info('Init validset and testset...')
  validset = DatasetObject2(config.keycode_validset_path, config.nl_validset_path)
  valid_loader = DataLoader(dataset=validset, batch_size=config.batch_size, shuffle=True,
      collate_fn=lambda *x: my_collate_fn2(x, keycode_vcblry, nl_vcblry))
  
  testset = DatasetObject2(config.keycode_testset_path, config.nl_testset_path)
  test_loader = DataLoader(dataset=testset, batch_size=config.batch_size, shuffle=False,
      collate_fn=lambda *x: my_collate_fn2(x, keycode_vcblry, nl_vcblry, is_raw_nl=True))
  logger.info('Done.')

  # 根据训练集生成词典
  logger.info('Build vocabularies...')
  keycode_vcblry = build_vcblry(trainset.keycode_set, 'keycode')
  nl_vcblry = build_vcblry(trainset.nl_set, 'nl')
  logger.info('Done.')

  keycode_vcblry_size = len(keycode_vcblry)
  nl_vcblry_size = len(nl_vcblry)

  logger.info('Training the pre-train model...')
  start_time = datetime.now()
  pre_train_model_state = train((keycode_vcblry_size, nl_vcblry_size), train_loader,
          valid_loader, nl_vcblry, len(trainset), is_main_model=False, logger=logger)
  logger.info('Training done. [Time taken: {!s}]'.format(datetime.now() - start_time))

  logger.info('Testing the pre-train model...')
  start_time = datetime.now()
  test(pre_train_model_state, (keycode_vcblry_size, nl_vcblry_size), test_loader,
        nl_vcblry, len(testset), is_main_model=False, logger=logger)
  logger.info('Testing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

