import argparse
class BaseOptions():
    def initialize(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--image_size', type=int, default=512)
        self.parser.add_argument('--batch_size', type=int, default=4)
        self.parser.add_argument('--result_path', type=str, default="./result")
        self.parser.add_argument('--experiment_name', type=str, default="exp")
        self.parser.add_argument('--device', type=str, default='cuda')
        self.parser.add_argument("--n_classes", type=int, default=1)

        self.parser.add_argument("--in_channels", type=int, default=3)
        return self.parser

    def get_opts(self):
        return self.parser.parse_args()