from preprocess import read_logo
from network import LogoWGan
import os
from utils import check_folder

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

check_folder('snapshot/')
check_folder('gen_logo/')
check_folder('saved_model/')

feed_dict = {
    'lr': 0.0001,
    'logo': read_logo(),
    'batch_size': 128,
    'iteration': 50000,
    'snapshot': 'snapshot/'
}


logo_gan = LogoWGan()
logo_gan.build_graph()
logo_gan.train(feed_dict)

