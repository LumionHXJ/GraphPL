from utils.BaseFlags import parser as parser

# DATA DEPENDENT
# to be set by experiments themselves
parser.add_argument('--clf_modality', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--clf_learning_rate', type=float, default=0.01, help="starting learning rate")
parser.add_argument('--clf_batchsize', type=int, default=256, help="starting learning rate")

parser.add_argument('--likelihood', type=str, default='laplace', help="output distribution")
parser.add_argument('--dim', type=int, default=32, help="dimension of latent")
