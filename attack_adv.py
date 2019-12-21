"""
Various adversaraial attacks for comparison on CelebA
"""

import argparse
import os
import numpy as np
import torch
import torchvision

from simple_classifier import Classifier, get_data_loader
from celebA_data_loader import CelebA_Dataset
from tqdm import tqdm

from advertorch import attacks
from advertorch.utils import predict_from_logits
from matplotlib import pyplot as plt
import pickle 

_ITER_ATTACKS = [attacks.LinfBasicIterativeAttack, attacks.LinfPGDAttack, attacks.MomentumIterativeAttack]

_SPECIAL_ATTACKS = [attacks.CarliniWagnerL2Attack]


class Attack(AttackConfig):


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir',
                        help='Path to data directory', required=True)
    parser.add_argument('-a', '--attrib_path',
                        help='Path to attrib file', required=True)
    parser.add_argument('-m', '--model_path',)
    parser.add_argument('--epochs', help='epochs', type=int,
                        required=False, default=10000)
    parser.add_argument('--train_attribute', required=True,
                        help='Attribute to train classifier on')
    parser.add_argument('-o', '--outdir', help='Output directory', default='./')
    parser.add_argument('--eps', help='Epsilon', type=float, default=0.3)
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.batch_size = 1
    model = Classifier(input_size=(256,256,3))
    model_state_d = torch.load(args.model_path, map_location={'cuda:0':'cpu'})

    model.load_state_dict(model_state_d)
    model.eval()
    device = torch.device('cuda')
    model.to(device)

    attack_vals = {}
    attack = _ITER_ATTACKS
    #for attack in _ITER_ATTACKS:
        #print(attack)
    attack_name = attack.__name__
    print(attack_name)
    attacker = attack(model, eps=args.eps, 
                        clip_min=-1.0, clip_max=1.0, targeted=False, nb_iter=100)
    attacker = attacker
    dl = get_data_loader(args, train='test', shuffle=False)
    success = 0
    idx = 0
    for (x, label) in tqdm(dl, total=500):
        idx+=1
        x = x.to(device)
        label=label.to(device)
        x_adv = attacker.perturb(x)
        #print(x_adv.size())
        #plt.imshow(x_adv.detach().cpu().numpy()[0,...].transpose(1,2,0))
        #plt.show()
        new_label = predict_from_logits(model(x_adv.to(device)))
        #print(new_label, label)
        if new_label != label:
            success+=1
        #print(idx)
        if idx == 500:
            print('Broken images: %d', success)
            break
    attack_vals[attack_name] = success
with open(os.path.join(args.outdir, '{}.pkl'.format(attack_name)),'wb') as f:
    pickle.dump(attack_vals, f)
    #cw_linf = attack.CarliniWagnerL2Attack()

if __name__ == '__main__':
    main()