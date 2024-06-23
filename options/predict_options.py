from options.base_options import BaseOptions


class PredictOptions(BaseOptions):
    def __init__(self):
        self.initialize()
    def initialize(self):
        self.parser=BaseOptions.initialize(self)
        self.parser.add_argument('--trained_checkpoint', type=str, default="./result/exp_2/checkpoints/epoch_50.pth")
        self.parser.add_argument('--image_folder', type=str, default='./datasets/CVC-300/images', help="image folder")
        return self.parser