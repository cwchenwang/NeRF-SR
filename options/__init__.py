import argparse

class Configurable(object):
    @staticmethod
    def modify_commandline_options(parser):
        """Add new options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser

        Returns:
            the modified parser.
        """
        return parser


def get_option_setter(obj):
    if issubclass(obj, Configurable):
        return obj.modify_commandline_options
    print('Class', obj, 'is not a subclass of Configurable, hence unable to modify commandline arguments.')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        