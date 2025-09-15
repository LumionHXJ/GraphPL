from torch.nn import functional as F
import torch.distributions as dist
from .lhoods import ComposeDistribution
import torch
import pickle
def output_demo_fn(x):
    # x: N, 20(8+5+7)
    return dist.OneHotCategorical(logits=x[..., :demographics_size['age']]), \
            dist.OneHotCategorical(logits=x[..., demographics_size['age']:demographics_size['age']+demographics_size['gender']]), \
            dist.OneHotCategorical(logits=x[..., -demographics_size['ethnicity']:])

def output_apache_fn(x):
    # x: N, 36
    pre = dist.Laplace(x[..., :3], torch.tensor(0.75).to(x.device))
    eyes = dist.OneHotCategorical(logits=x[..., 3:3+APACHEAPSVAR_NCAT['eyes']])
    motor = dist.OneHotCategorical(logits=x[..., 3+APACHEAPSVAR_NCAT['eyes']:3+APACHEAPSVAR_NCAT['eyes']+APACHEAPSVAR_NCAT['motor']])
    verbal = dist.OneHotCategorical(logits=x[..., 3+APACHEAPSVAR_NCAT['eyes']+APACHEAPSVAR_NCAT['motor']:18])
    post = dist.Laplace(x[..., 18:], torch.tensor(0.75).to(x.device))
    return pre, eyes, motor, verbal, post

def output_lab_fn(x):
    return x, torch.tensor(0.75).to(x.device)

def output_code_fn(x):
    return (torch.sigmoid(x), )

APACHEAPSVAR = [
    "intubated",
    "vent",
    "dialysis",
    "eyes",
    "motor",
    "verbal",
    "meds",
    "urine",
    "wbc",
    "temperature",
    "respiratoryrate",
    "sodium",
    "heartrate",
    "meanbp",
    "ph",
    "hematocrit",
    "creatinine",
    "albumin",
    "pao2",
    "pco2",
    "bun",
    "glucose",
    "bilirubin",
    "fio2",
]

APACHEAPSVAR_NCAT = {
    "eyes": 4,
    "motor": 6,
    "verbal": 5,
}


modalities = ['demographics', 'lab', 'apache', 'diagnosis', 'treatment', 'medication']
demographics_size = {"age": 8, 'gender': 5, 'ethnicity': 7}
modality_dims = {'demographics': 256, 'lab': 256, 'apache': 256, 'diagnosis': 256, 'treatment': 256, 'medication': 256}
ffn_layers = {'demographics': 2, 'lab': 3, 'apache': 2, 'diagnosis': 3, 'treatment': 3, 'medication': 3}
input_dim = {'demographics': 20, 'lab': 158, 'apache': 36, 'diagnosis': 76, 'treatment': 272, 'medication': 282}
lhoods = {'demographics': ComposeDistribution, 'lab': 'laplace', 'apache': ComposeDistribution, 
          'diagnosis': 'bernoulli', 'treatment': 'bernoulli', 'medication': 'bernoulli'}

# right after linear
output_fn = {'demographics': output_demo_fn, 'lab': output_lab_fn, 
             'apache': output_apache_fn, 'diagnosis': output_code_fn, 
             'treatment': output_code_fn, 'medication': output_code_fn}

with open('muse_eicu/alpha.pkl', mode='rb') as f:
    alpha = pickle.load(f)