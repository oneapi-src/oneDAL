# file: impl_als_csr_batch.py
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

## <a name="DAAL-EXAMPLE-PY-IMPLICIT_ALS_CSR_BATCH"></a>
## \example impl_als_csr_batch.py

import os
import sys

import daal.algorithms.implicit_als.prediction.ratings as ratings
import daal.algorithms.implicit_als.training as training
import daal.algorithms.implicit_als.training.init as init

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable, createSparseTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
trainDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'implicit_als_csr.csv')

# Algorithm parameters
nFactors = 2

dataTable = None
initialModel = None
trainingResult = None


def initializeModel():
    global initialModel, dataTable

    # Read trainDatasetFileName from a file and create a numeric table to store the input data
    dataTable = createSparseTable(trainDatasetFileName)

    # Create an algorithm object to initialize the implicit ALS model with the default method
    initAlgorithm = init.Batch(method=init.fastCSR)
    initAlgorithm.parameter.nFactors = nFactors

    # Pass a training data set and dependent values to the algorithm
    initAlgorithm.input.set(init.data, dataTable)

    # Initialize the implicit ALS model
    res = initAlgorithm.compute()
    # (Result class from implicit_als.training.init)
    initialModel = res.get(init.model)


def trainModel():
    global trainingResult

    # Create an algorithm object to train the implicit ALS model with the default method
    algorithm = training.Batch(method=training.fastCSR)

    # Pass a training data set and dependent values to the algorithm
    algorithm.input.setTable(training.data, dataTable)
    algorithm.input.setModel(training.inputModel, initialModel)

    algorithm.parameter.nFactors = nFactors

    # Build the implicit ALS model
    # Retrieve the algorithm results
    trainingResult = algorithm.compute()


def testModel():

    # Create an algorithm object to predict recommendations of the implicit ALS model
    algorithm = ratings.Batch()
    algorithm.parameter.nFactors = nFactors

    algorithm.input.set(ratings.model, trainingResult.get(training.model))

    res = algorithm.compute()

    predictedRatings = res.get(ratings.prediction)

    printNumericTable(predictedRatings, "Predicted ratings:")

if __name__ == "__main__":

    initializeModel()
    trainModel()
    testModel()
