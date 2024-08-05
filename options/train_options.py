from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        self.initialize()
    def initialize(self):
        self.parser=BaseOptions.initialize(self)
        self.parser.add_argument('--weight_decay', type=float, default=0)
        self.parser.add_argument('--epochs', type=int, default=50)
        self.parser.add_argument("--lr", type=float, default=1e-4)
        self.parser.add_argument('--root_path', type=str, default='./datasets/CVC-300', help="dataset root")
        self.parser.add_argument('--test_root_path', type=str, default='./datasets/CVC-300', help="test dataset root")
        self.parser.add_argument('--subdir', type=str, nargs="+", default=["images", "masks"])
        self.parser.add_argument('--checkpoint_save_freq', type=int, default=5, help="checkpoint save frequency")
        self.parser.add_argument('--seed', type=int, default=1234, help="random seed")
        return self.parser

