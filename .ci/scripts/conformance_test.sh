#!/bin/bash

#!/bin/bash
conda install -y -c conda-forge pytest scikit-learn=0.21
cd .ci/scripts/conformance-scripts/
python run_tests.py
cd ../../..
