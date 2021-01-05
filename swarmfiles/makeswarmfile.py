import glob
import os

# Create list of ROIs we want that are found in data_path
data_path = '../data/'
rois = []

# get US rois
for name in glob.glob(data_path + 'covidtimeseries_US_??.csv'):
    roi = os.path.basename(name)
    rois.append(roi[16:-4])

# add Canadian provinces
canadian_provinces = ['Alberta', 'BC', 'Manitoba', 'New Brunswick', 'NL',
                'Nova Scotia', 'Nunavut', 'NWT', 'Ontario', 'PEI', 'Quebec',
                'Saskatchewan', 'Yukon']
for prov in canadian_provinces:
    prov = 'CA_' + prov
    rois.append(prov)

rois.sort()

models = ['SICRdiscrete', 'SICRdiscrete1', 'SICRMQC2R2DX']


for model in models:
    swarmFile = open(f"../{model}.swarm", "w")
    for roi in rois:
        line = ("source /data/schwartzao/conda/etc/profile.d/conda.sh "
            "&& conda activate covid "
            f"&& python /home/schwartzao/covid-sicr/scripts/run.py {model} "
            f"-r='{roi}' "
            "-mp='/home/schwartzao/covid-sicr/models/' "
            "-fp='/data/schwartzao/covid-sicr/fits20210105/' " # changes
            "-it=10000	-wm=6000	-f=1	-ad=.85	-ft=1"
            )
        swarmFile.write(line)
        swarmFile.write('\n')
    swarmFile.close()
