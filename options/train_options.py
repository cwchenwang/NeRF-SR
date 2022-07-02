from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        parser.set_defaults(phase='train')

        # visdom and HTML visualization parameters
        parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        parser.add_argument('--vis_freq', type=int, default=10)
        parser.add_argument('--vis_epoch_freq', type=int, default=5)
        parser.add_argument('--val_freq', type=int, default=10)
        parser.add_argument('--val_epoch_freq', type=int, default=1)

        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--init_weights', type=str, default=None, help='initialize weights from an existing model, in format [name]:[epoch]')
        parser.add_argument('--init_weights_keys', type=str, default='.+', help='regex for weights keys to be loaded')

        parser.add_argument('--fix_layers', type=str, default=None, help='regex for fix layers')

        # training parameters
        parser.add_argument('--precrop_frac', type=float, default=0.5, help='center crop factor')
        parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs with the initial learning rate')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='exp', help='learning rate policy. [linear | exp | step]')

        # linear | exp policy
        parser.add_argument('--lr_final', type=float, default=5e-6, help='final learning rate for adam, used in linear and exp')
        parser.add_argument('--n_epochs_decay', type=int, default=10, help='number of epochs to decay learning rate to lr_final')

        # step policy
        parser.add_argument('--lr_decay_epochs', type=int, default=10, help='multiply by a gamma every lr_decay_iters epochs, used in step')
        parser.add_argument('--lr_decay_gamma', type=float, default=0.1, help='multiply by a gamma every lr_decay_iters epochs, used in step')

        parser.add_argument('--keep_last', action='store_true')

        # data split
        parser.add_argument('--train_split', type=str, default='train')
        parser.add_argument('--val_split', type=str, default='val')
        parser.add_argument('--val_epoch_split', type=str, default='test_train')
        parser.add_argument('--test_split', type=str, default='test')

        # gradient clip
        parser.add_argument('--grad_clip_val', type=float, default=0)
        parser.add_argument('--grad_clip_type', type=str, default='norm', choices=['norm', 'value'])

        self.isTrain, self.isTest, self.isInfer = True, False, False
        return parser
