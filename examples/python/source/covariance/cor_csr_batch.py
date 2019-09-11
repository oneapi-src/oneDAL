# file: cor_csr_batch.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

## <a name="DAAL-EXAMPLE-PY-CORRELATION_CSR_BATCH"></a>
## \example cor_csr_batch.py

import os
import sys

from daal.algorithms import covariance

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable, createSparseTable

DAAL_PREFIX = os.path.join('..', 'data')

#  Input matrix is stored in one-based sparse row storage format
datasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'covcormoments_csr.csv')

if __name__ == "__main__":

    # Read datasetFileName from file and create numeric table for storing input data
    dataTable = createSparseTable(datasetFileName)

    # Create algorithm to compute correlation matrix using default method
    algorithm = covariance.Batch()
    algorithm.input.set(covariance.data, dataTable)

    # Set the parameter to choose the type of the output matrix
    algorithm.parameter.outputMatrixType = covariance.correlationMatrix

    # Get computed correlation
    res = algorithm.compute()

    printNumericTable(res.get(covariance.correlation), "Correlation matrix (upper left square 10*10) :", 10, 10)
    printNumericTable(res.get(covariance.mean),        "Mean vector:", 1, 10)
