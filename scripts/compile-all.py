"""Fit many models for many regions.  If you have HPC access it is recommended
to instead use `run.py` in parallel."""

import argparse
from pathlib import Path
from tqdm.auto import tqdm

import niddk_covid_sicr as ncs

# Parse all the command-line arguments
parser = argparse.ArgumentParser(description='Force compile all Stan models')

parser.add_argument('-mn', '--model-names', default=[], nargs='+',
                    help='Name of the Stan model file (without extension)')
parser.add_argument('-mp', '--models-path', default='./models',
                    help='Path to directory containing the .stan model files')
parser.add_argument('-fc', '--force-compile', default=1,
                    help='Force compile all models (no reliance on previous compilation)')
args = parser.parse_args()

if not args.model_names:
    args.model_names = ncs.list_models(args.models_path)
    assert len(args.model_names),\
        ("No such model files matching: *.stan' at %s" % (args.models_path))

model_paths = [Path(args.models_path) / ('%s.stan' % model_name)
               for model_name in args.model_names]
for model_path in model_paths:
    assert model_path.is_file(), "No such .stan file: %s" % model_path

run_script_path = Path(__file__).parent / 'run.py'
run_flags = [('--%s' % key.replace('_', '-'), value)
             for key, value in args.__dict__.items()
             if key not in ['model_names', 'rois']]

# This next section will be run in serial since this is only a reference
# implementation. Parallel implementations are possible with multiprocessing
# or the library of your choice. Best performance comes from parallelizing
# run.py on a cluster.
iterator = tqdm(args.model_names, desc='Model')
for model_name in iterator:
    iterator.set_description("Compiling %s" % model_name)
    ncs.load_or_compile_stan_model(model_name, models_path=args.models_path,
                                   force_recompile=True)

print("Finished compiling all models")
