from .base_options import BaseOptions
from datetime import datetime


class InferOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options

        parser.set_defaults(phase='infer')

        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='infer', help='train, test, infer')
        parser.add_argument('--data_name', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"))

        self.isTrain, self.isTest, self.isInfer = False, False, True
        return parser
