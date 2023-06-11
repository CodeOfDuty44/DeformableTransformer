import argparse
from config import get_config
# from models.dat import build_model
import torch
from models.dat import DAT
import torchvision.datasets as datasets

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="/Users/serkan/Desktop/DAT/configs/dat_tiny.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # easy config modification
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--pretrained', type=str, help='Finetune 384 initial checkpoint.', default='')

    args = parser.parse_args()

    config = get_config(args)

    return args, config

def main():
    dataset = datasets.ImageNet(root='path/to/imagenet', split='train')
    data
    x = dataset.__getitem__(0)

    model = DAT(224, 1000)
    x = torch.zeros(5,3,224,224).to()
    x.requires_grad = True
    out = model(x).to()
    44

if __name__ == "__main__":
    main()