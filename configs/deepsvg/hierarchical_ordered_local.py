from .default_icons import *


class ModelConfig(Hierarchical):
    def __init__(self):
        super().__init__()

        self.label_condition = False
        self.use_vae = False


class Config(Config):
    def __init__(self, num_gpus=2):
        super().__init__(num_gpus=num_gpus)

        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        self.filter_category = None

        self.learning_rate = 1e-3 * num_gpus
        self.batch_size = 60 * num_gpus

        self.num_epochs = 1
        self.train_ratio = 0.999
        self.val_every = 1
        self.batch_size = 20
        self.pretrained_path = "pretrained/hierarchical_ordered.pth.tar"
