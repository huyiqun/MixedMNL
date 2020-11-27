import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-v", "--verbose", action="count", default=3, help="verbose level")
parser.add_argument(
    "--product_data", type=str, help="product data", default="single.ps"
)
parser.add_argument(
    "--consumer_data", type=str, help="consumer data", default="single.pop"
)
parser.add_argument("--num_type", type=int, help="number of consumer types", default=2)
parser.add_argument("--num_prod", type=int, help="number of products", default=2)
parser.add_argument("--num_feat", type=int, help="number of features", default=1)
parser.add_argument("--simulate", help="simulate data", action="store_true")
parser.add_argument("--save_data", help="save simulated data", action="store_true")
parser.set_defaults(
    simulate=False,
    product_data="data/product.csv",
    consumer_data="data/consumer.csv",
    write=False,
)

parser.add_argument(
    "--exp_per_cyc",
    type=int,
    help="number of independent experiments per cycle",
    default=5,
)
parser.add_argument(
    "--sim_per_exp", type=int, help="number of simulations per experiment", default=10
)
parser.add_argument(
    "--trans_per_exp",
    type=int,
    help="number of transactions per experiment",
    default=1000,
)
parser.add_argument("--sample_ratio", type=float, help="sample ratio", default=0.2)


args = parser.parse_args()

CYCLE_SEP = "=" * 80 + "\n"
EXP_SEP = "~" * 80 + "\n"

VERBOSE = args.verbose
CONSUMER_DATA = args.consumer_data
PRODUCT_DATA = args.product_data

SIMULATE = args.simulate
SAVE_DATA = args.save_data
NUM_TYPE = args.num_type
NUM_PRODUCT = args.num_prod
NUM_FEATURE = args.num_feat

EXP_PER_CYC = args.exp_per_cyc
SIM_PER_EXP = args.sim_per_exp
TRANS_PER_EXP = args.trans_per_exp
SAMPLE_RATIO = args.sample_ratio

if not SIMULATE:
    assert CONSUMER_DATA is not None, "need consumer data when not simulating"
    assert PRODUCT_DATA is not None, "need product data when not simulating"
