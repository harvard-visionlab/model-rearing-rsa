import torch
from collections import defaultdict
from functools import wraps

from model_rearing_workshop.models import load_model_from_weights
from model_rearing_workshop.models.weights import Weights, get_standard_transforms

alexnet2023_w1_barlow = Weights(
    url='https://visionlab-members.s3.wasabisys.com/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/barlow/20231116_044210/final_weights-0b70b9da61.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/barlow/20231116_044210/params-0b70b9da61.json',
            train='https://visionlab-members.s3.wasabisys.com/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/barlow/20231116_044210/log_train-0b70b9da61.txt',
            val='https://visionlab-members.s3.wasabisys.com/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/barlow/20231116_044210/log_val-0b70b9da61.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

alexnet2023_w1_simclr =  Weights(
    url='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/simclr/20231114_102214/final_weights-150833a226.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/simclr/20231114_102214/params-150833a226.json',
            train='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/simclr/20231114_102214/log_train-150833a226.txt',
            val='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/simclr/20231114_102214/log_val-150833a226.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

alexnet2023_w1_supervised = Weights(
    url='https://visionlab-members.s3.wasabisys.com/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/supervised/20231115_044726/final_weights-d9ebe67438.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/supervised/20231115_044726/params-d9ebe67438.json',
            train='https://visionlab-members.s3.wasabisys.com/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/supervised/20231115_044726/log_train-d9ebe67438.txt',
            val='https://visionlab-members.s3.wasabisys.com/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/supervised/20231115_044726/log_val-d9ebe67438.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

alexnet2023_w1_vicreg = Weights(
    url='https://visionlab-members.s3.wasabisys.com/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/vicreg/20231116_094430/final_weights-d95d3b7cba.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/vicreg/20231116_094430/params-d95d3b7cba.json',
            train='https://visionlab-members.s3.wasabisys.com/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/vicreg/20231116_094430/log_train-d95d3b7cba.txt',
            val='https://visionlab-members.s3.wasabisys.com/alvarez/Projects/model_rearing_workshop/models/in1k/alexnet2023_baseline/vicreg/20231116_094430/log_val-d95d3b7cba.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

alexnet2023_w2_barlow = Weights(
    url='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/barlow/20231122_201921/final_weights-ff6c4655f2.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/barlow/20231122_201921/params-ff6c4655f2.json',
            train='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/barlow/20231122_201921/log_train-ff6c4655f2.txt',
            val='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/barlow/20231122_201921/log_val-ff6c4655f2.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

alexnet2023_w2_simclr = Weights(
    url='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/simclr/20231121_113023/final_weights-67998b280c.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/simclr/20231121_113023/params-67998b280c.json',
            train='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/simclr/20231121_113023/log_train-67998b280c.txt',
            val='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/simclr/20231121_113023/log_val-67998b280c.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)


alexnet2023_w2_supervised = Weights(
    url='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/supervised/20231121_121633/final_weights-c9ccb94b09.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/supervised/20231121_121633/params-c9ccb94b09.json',
            train='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/supervised/20231121_121633/log_train-c9ccb94b09.txt',
            val='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/supervised/20231121_121633/log_val-c9ccb94b09.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

alexnet2023_w2_vicreg = Weights(
    url='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/vicreg/20231125_191933/final_weights-4ce75725a8.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/vicreg/20231125_191933/params-4ce75725a8.json',
            train='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/vicreg/20231125_191933/log_train-4ce75725a8.txt',
            val='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w2/vicreg/20231125_191933/log_val-4ce75725a8.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)


alexnet2023_w3_barlow = Weights(
    url='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/barlow/20231122_202537/final_weights-01e9a08392.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/barlow/20231122_202537/params-01e9a08392.json',
            train='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/barlow/20231122_202537/log_train-01e9a08392.txt',
            val='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/barlow/20231122_202537/log_val-01e9a08392.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

alexnet2023_w3_simclr = Weights(
    url='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/simclr/20231121_210405/final_weights-9325e1684d.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/simclr/20231121_210405/params-9325e1684d.json',
            train='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/simclr/20231121_210405/log_train-9325e1684d.txt',
            val='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/simclr/20231121_210405/log_val-9325e1684d.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

alexnet2023_w3_supervised = Weights(
    url='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/supervised/20231121_210219/final_weights-7d0fd2349a.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/supervised/20231121_210219/params-7d0fd2349a.json',
            train='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/supervised/20231121_210219/log_train-7d0fd2349a.txt',
            val='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/supervised/20231121_210219/log_val-7d0fd2349a.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

alexnet2023_w3_vicreg = Weights(
    url='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/vicreg/20231126_070309/final_weights-59520f7c43.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/vicreg/20231126_070309/params-59520f7c43.json',
            train='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/vicreg/20231126_070309/log_train-59520f7c43.txt',
            val='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w3/vicreg/20231126_070309/log_val-59520f7c43.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

alexnet2023_w4_barlow = Weights(
    url='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/barlow/20231122_202821/final_weights-fb283eab30.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/barlow/20231122_202821/params-fb283eab30.json',
            train='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/barlow/20231122_202821/log_train-fb283eab30.txt',
            val='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/barlow/20231122_202821/log_val-fb283eab30.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

alexnet2023_w4_simclr = Weights(
    url='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/simclr/20231122_065601/final_weights-0ff9ae8dab.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/simclr/20231122_065601/params-0ff9ae8dab.json',
            train='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/simclr/20231122_065601/log_train-0ff9ae8dab.txt',
            val='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/simclr/20231122_065601/log_val-0ff9ae8dab.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

alexnet2023_w4_supervised = Weights(
    url='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/supervised/20231122_064009/final_weights-b26b882420.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/supervised/20231122_064009/params-b26b882420.json',
            train='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/supervised/20231122_064009/log_train-b26b882420.txt',
            val='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/supervised/20231122_064009/log_val-b26b882420.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

alexnet2023_w4_vicreg = Weights(
    url='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/vicreg/20231126_072810/final_weights-b15b986e06.pth',
    transforms=get_standard_transforms(), # Add your transforms here
    meta={
        "repo": "https://github.com/harvard-visionlab/alexnets",
        "urls": dict(
            params='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/vicreg/20231126_072810/params-b15b986e06.json',
            train='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/vicreg/20231126_072810/log_train-b15b986e06.txt',
            val='https://visionlab-members.s3.wasabisys.com/tkonkle/Projects/model_rearing_workshop/models/in1k/alexnet2023_w4/vicreg/20231126_072810/log_val-b15b986e06.txt',
        ),
        "_metrics": {},
        "_docs": """
            ....
        """,
    },
)

model_weights = defaultdict(dict)
model_weights['w1']['barlow'] = alexnet2023_w1_barlow
model_weights['w1']['simclr'] = alexnet2023_w1_simclr
model_weights['w1']['supervised'] = alexnet2023_w1_supervised
model_weights['w1']['vicreg'] = alexnet2023_w1_vicreg

model_weights['w2']['barlow'] = alexnet2023_w2_barlow
model_weights['w2']['simclr'] = alexnet2023_w2_simclr
model_weights['w2']['supervised'] = alexnet2023_w2_supervised
model_weights['w2']['vicreg'] = alexnet2023_w2_vicreg

model_weights['w3']['barlow'] = alexnet2023_w3_barlow
model_weights['w3']['simclr'] = alexnet2023_w3_simclr
model_weights['w3']['supervised'] = alexnet2023_w3_supervised
model_weights['w3']['vicreg'] = alexnet2023_w3_vicreg

model_weights['w4']['barlow'] = alexnet2023_w4_barlow
model_weights['w4']['simclr'] = alexnet2023_w4_simclr
model_weights['w4']['supervised'] = alexnet2023_w4_supervised
model_weights['w4']['vicreg'] = alexnet2023_w4_vicreg

def load_model(width="w2", task="supervised", device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = model_weights[width][task]
    model = load_model_from_weights(weights)
    model.to(device)
    return model