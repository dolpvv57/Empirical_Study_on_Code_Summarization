# 使用关键代码行序列进行预训练
import torch
import torch.nn as nn
import config
from .encoder import Encoder
from .decoder import Decoder
import random

class PreTrainModel(nn.Module):
  def __init__(self, keycode_vcblry_size, nl_vcblry_size, load_path=None,
            load_dict=None, is_eval=False):
    super(PreTrainModel, self).__init__()
    self.nl_vcblry_size = nl_vcblry_size
    self.is_eval = is_eval
    self.keycode_encoder = Encoder(keycode_vcblry_size)
    self.decoder = Decoder(self.nl_vcblry_size, is_pretrain=True)

    if config.use_cuda:
      self.keycode_encoder = self.keycode_encoder.cuda(config.cuda_id)
      self.decoder = self.decoder.cuda(config.cuda_id)
    
    if load_path or load_dict:
      state_dict = torch.load(load_path) if not load_dict else load_dict

      self.keycode_encoder.load_state_dict(state_dict['keycode_encoder'])
      self.decoder.load_state_dict(state_dict['decoder'])

    if self.is_eval:
      self.keycode_encoder.eval()
      self.decoder.eval()

  def forward(self, batch_data, nl_bos_idx, is_test=False):
    keycode_batch_data, keycode_seq_lens, _, _, nl_batch_data, nl_seq_lens = batch_data
    keycode_enc_opt, keycode_enc_hdn = self.keycode_encoder(keycode_batch_data, keycode_seq_lens)
    last_dec_hdn = keycode_enc_hdn[:1]

    if is_test:
      return keycode_enc_opt, last_dec_hdn
    
    max_dec_step = max(nl_seq_lens)
    cur_batch_size = len(keycode_seq_lens)
    dec_input = torch.tensor([nl_bos_idx]*cur_batch_size, device=config.device)
    dec_output = torch.zeros((max_dec_step, cur_batch_size, self.nl_vcblry_size), device=config.device)

    for cur_step in range(max_dec_step):
      cur_dec_output, last_dec_hdn = self.decoder(dec_input, last_dec_hdn, keycode_enc_opt)
      dec_output[cur_step] = cur_dec_output

      if config.use_teacher_forcing and random.random() < config.teacher_forcing_ratio and not self.is_eval:
        dec_input = nl_batch_data[cur_step]
      else:
        _, indices = cur_dec_output.topk(1)
        dec_input = indices.squeeze(1).detach().to(config.device)

    return dec_output
