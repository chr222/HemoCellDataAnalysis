#!/bin/bash
if [[ "$0" == "$BASH_SOURCE" ]]; then
 echo "Script is a subshell, this wont work, source it instead!"
 exit 1
fi

module load cray-hdf5/1.12.0.7
module load cray-python/3.9.4.1
module list

export PYTHONUSERBASE=/work/e723/e723/cspieke/.local
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9.4.1/site-packages:$PYTHONPATH

/opt/cray/pe/python/3.9.4.1/bin/python3.9 -m pip install --upgrade pip
pip install --user h5py~=3.2.1
pip install --user -U numpy
pip install --user setuptools~=56.0.0
pip install --user matplotlib
pip install --user scipy
pip install --user pandas
pip install --user psutil
pip install --user pytest
pip install --user beautifulsoup4
pip install --user lxml