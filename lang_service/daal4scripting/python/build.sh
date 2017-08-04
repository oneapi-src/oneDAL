#!/bin/bash

env CNCROOT=${PREFIX} TBBROOT=${PREFIX} DAALROOT=${PREFIX} ${PYTHON} setup.py install
