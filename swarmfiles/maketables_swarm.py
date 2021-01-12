import argparse
import glob
from pathlib import Path
import os

# Parse all the command-line arguments
parser = argparse.ArgumentParser(description=('Creates swarm file for '
                                              'make-tables.py'))

parser.add_argument('-fn', '--file-name', default='maketables',
                    help='Swarm filename')
parser.add_argument('-fp', '--fits-path', default='/data/schwartzao/covid-sicr/fits',
                    help='Path to directory to save fit files')
parser.add_argument('-tp', '--tables-path', default='/data/schwartzao/covid-sicr/tables/',
                    help='Path to directory to save tables')

args = parser.parse_args()

swarmFile = open(f"./{args.file_name}.swarm", "w")
line = ("source /data/schwartzao/conda/etc/profile.d/conda.sh "
        "&& conda activate covid "
        "&& python /home/schwartzao/covid-sicr/scripts/make-tables.py "
        f"-fp='{args.fits_path}' "
        f"-tp='{args.tables_path}' "
        )
print(line)

swarmFile.write(line)
swarmFile.write('\n')
swarmFile.close()
