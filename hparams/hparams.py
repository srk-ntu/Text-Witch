from .registry import register


class HParams():

  def __init__(self):
    self.batch_size = 32
    self.num_epochs = 256
    self.lr = 1e-3
    self.save_every = 5000
    self.test_every = 100
    self.eval_steps = 10


@register
def spam_data_cnn():
  hps = HParams()
  hps.data_dir = [
      '/home/srk/NTU/Text-Witch/data/Spam_Detection_Data/truthful_pos',
      '/home/srk/NTU/Text-Witch/data/Spam_Detection_Data/truthful_neg',
      '/home/srk/NTU/Text-Witch/data/Spam_Detection_Data/deceptive_pos',
      '/home/srk/NTU/Text-Witch/data/Spam_Detection_Data/deceptive_neg'
  ]
  hps.embedding_dim = 128
  hps.max_length = 784
  hps.num_classes = 2
  hps.filter_sizes = [3, 4, 5]
  hps.num_filters = [128, 128, 128]
  hps.model = "CNN"
  hps.drop_rate = 0.5

  return hps