import torch

# sbt的输入层里面输入code 相当于code+keycode同时输入
dataset_base_path = '../data/tl-codesum/'
keycode_trainset_path = dataset_base_path + 'train/code.body.rep'
sbt_trainset_path = dataset_base_path + 'train/sbt.seq'
nl_trainset_path = dataset_base_path + 'train/nl.original'

keycode_validset_path = dataset_base_path + 'valid/code.body.rep'
sbt_validset_path = dataset_base_path + 'valid/sbt.seq'
nl_validset_path = dataset_base_path + 'valid/nl.original'

keycode_testset_path = dataset_base_path + 'test/code.body.rep'
sbt_testset_path = dataset_base_path + 'test/sbt.seq'
nl_testset_path = dataset_base_path + 'test/nl.original'

log_name = 'code_body_rep'
out_name = 'code_body_rep'

max_keycode_length = 500
max_sbt_length = 500
max_nl_length = 30
max_translate_length = 30
batch_size = 64

max_vcblry_size = 50000
vcblry_base_path = '{}vcblry/'.format(dataset_base_path)

encoder_hdn_size = 256
decoder_hdn_size = 256
embedding_dim = 256
use_cuda = torch.cuda.is_available()
cuda_id = 1
device = torch.device("cuda:{}".format(cuda_id) if torch.cuda.is_available() else "cpu") 
use_teacher_forcing = True
teacher_forcing_ratio = 0.5
learning_rate = 0.001
lr_decay_every = 1
lr_decay_rate = 0.95

model_base_path = '{}trained_model/'.format(dataset_base_path)
load_pre_train_model = False
pretrain_early_stopping_rounds = 5
main_early_stopping_rounds = 5
num_epoch = 100
print_every_batch_num = 200
pretrain_valid_every_iter = 5000

beam_width = 5
rand_seed = 28