
from utils.BaseFlags import parser as parser

# add arguments
parser.add_argument('--len_sequence', type=int, default=256, help="length of sequence")
parser.add_argument('--img_size', type=int, default=64, help="img dimension (width/height)")
parser.add_argument('--image_channels', type=int, default=3, help="number of channels in images")
parser.add_argument('--crop_size_img', type=int, default=148, help="number of channels in images")
parser.add_argument('--dir_text', type=str, default='../text', help="directory where text is stored")
parser.add_argument('--random_text_ordering', type=bool, default=False,
                    help="flag to indicate if attributes are shuffled randomly")
parser.add_argument('--random_text_startindex', type=bool, default=True,
                    help="flag to indicate if start index is random")

parser.add_argument('--DIM_text', type=int, default=64, help="filter dimensions of residual layers")
parser.add_argument('--DIM_img', type=int, default=64, help="filter dimensions of residual layers")
parser.add_argument('--num_layers_text', type=int, default=7, help="number of residual layers")
parser.add_argument('--num_layers_img', type=int, default=5, help="number of residual layers")
parser.add_argument('--likelihood_m1', type=str, default='laplace', help="output distribution")
parser.add_argument('--likelihood_m2', type=str, default='categorical', help="output distribution")
parser.add_argument('--likelihood_m3', type=str, default='laplace', help="output distribution")
parser.add_argument('--likelihood_m4', type=str, default='laplace', help="output distribution")

#weighting of loss terms
parser.add_argument('--div_weight_m1_content', type=float, default=0.2, help="default weight divergence term content modality 1")
parser.add_argument('--div_weight_m2_content', type=float, default=0.2, help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_m3_content', type=float, default=0.2, help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_m4_content', type=float, default=0.2, help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_uniform_content', type=float, default=0.2, help="default weight divergence term prior")