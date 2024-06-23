from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        self.initialize()
    def initialize(self):
        self.parser=BaseOptions.initialize(self)
        self.parser.add_argument('--trained_checkpoint', type=str, default="./result/exp_2/checkpoints/epoch_50.pth")
        self.parser.add_argument('--test_root_path', type=str, default='./datasets/CVC-300', help="test dataset root")
        self.parser.add_argument('--subdir', type=str, nargs="+", default=["images", "masks"])
        return self.parser