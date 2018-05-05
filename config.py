import torch

# file_name = "./SAD.csv"
file_name = "./sentiment-analysis-dataset.csv"

# data config
MIN_FREQ = 10
MAX_VOCAB_SIZE = 40000
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 128
TRAIN_RATIO = 0.7
TEST_RATIO = 0.3

# device config
CUDA_VISIBLE_DEVICES = 0  # 可用显卡编号
torch.cuda.set_device(CUDA_VISIBLE_DEVICES)

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

# model config
DIM = 300  # embedding dims
HIDDEN_SIZE = 128
NUM_LAYER = 1
drop_out = 0.5
EPOCH = 5

# cnn config
kernel_sizes = (3, 4, 5)
