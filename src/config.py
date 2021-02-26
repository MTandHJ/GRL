







# Here are some basic settings.
# It could be overwritten if you want to specify
# specific configs. However, please check the corresponding
# codes in loadopts.py.



import torchvision.transforms as T
import random
from .dict2obj import Config



ROOT = "../data"
INFO_PATH = "./infos/{method}/{dataset}-{model}/{description}"
LOG_PATH = "./logs/{method}/{dataset}-{model}/{description}"
TIMEFMT = "%m%d%H"

TRANSFORMS = {
    "mnist": {
        'default': T.ToTensor()
    },
    "cifar10": {
        'default': T.Compose((
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ))
    }
}
TRANSFORMS['cifar100'] = TRANSFORMS['cifar10']

VALIDER = {
    "mnist": (Config(attack_type="pgd-linf", stepsize=0.033333, steps=100), 0.3),
    "cifar10": (Config(attack_type="pgd-linf", stepsize=0.1, steps=20), 8/255),
    "cifar100": (Config(attack_type="pgd-linf", stepsize=0.1, steps=20), 8/255)
}
BOUNDS = (0, 1)
# env settings
NUM_WORKERS = 3
PIN_MEMORY = True

# basic properties of inputs
MEANS = {
    "mnist": None,
    "cifar10": [0.4914, 0.4824, 0.4467],
    "cifar100": [0.5071, 0.4867, 0.4408],
}

STDS = {
    "mnist": None,
    "cifar10": [0.2471, 0.2435, 0.2617],
    "cifar100": [0.2675, 0.2565, 0.2761],
}

# the settings of optimizers of which lr could be pointed
# additionally.
OPTIMS = {
    "sgd": Config(lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=False),
    "adam": Config(lr=0.01, betas=(0.9, 0.999), weight_decay=0.)
}

VALIDER = {
    "mnist": (Config(attack_type="pgd", stepsize=0.033333, steps=100), 0.3),
    "cifar10": (Config(attack_type="pgd", stepsize=0.1, steps=20), 8/255),
    "cifar100": (Config(attack_type="pgd", stepsize=0.1, steps=20), 8/255)
}

# the learning schedular can be added here
LEARNING_POLICY = {
    "default": (
        "MultiStepLR",
        Config(
            milestones=[50, 75],
            gamma=0.1
        ),
        "Default leaning policy will be applied: " \
        "decay the learning rate at 50 and 75 epochs by a factor 10."
    ),
   "cosine":(   
        "CosineAnnealingLR",   
        Config(          
            T_max=200,
            eta_min=0.,
            last_epoch=-1,
        ),
        "cosine learning policy: T_max == epochs - 1"
    )
}




