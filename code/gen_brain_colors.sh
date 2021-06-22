#~/bin/bash

module load anaconda
source activate torch_env

for lag in -1500 -1200 -900 -600 -300 0 300 600 900 1200 1500; d
		--lag "$lag"
done
