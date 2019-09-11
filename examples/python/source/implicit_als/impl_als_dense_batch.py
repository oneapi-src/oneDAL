# file: impl_als_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-IMPLICIT_ALS_DENSE_BATCH"></a>
## \example impl_als_dense_batch.py

import os
import sys

import daal.algorithms.implicit_als.training.init
import daal.algorithms.implicit_als.prediction.ratings
from daal.algorithms.implicit_als import training, prediction
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
trainDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'implicit_als_dense.csv')

# Algorithm parameters
nFactors = 2

dataTable = None
initialModel = None
trainingResult = None


def initializeModel():
    global dataTable, initialModel

    # Read trainDatasetFileName from a file and create a numeric table to store the input data
    dataSource = FileDataSource(
        trainDatasetFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the input data
    dataSource.loadDataBlock()

    dataTable = dataSource.getNumericTable()
    # Create an algorithm object to initialize the implicit ALS model with the default method
    initAlgorithm = training.init.Batch()
    initAlgorithm.parameter.nFactors = nFactors

    # Pass a training data set and dependent values to the algorithm
    initAlgorithm.input.set(training.init.data, dataTable)
    res = initAlgorithm.compute()

    # Initialize the implicit ALS model
    initialModel = res.get(training.init.model)


def trainModel():
    global trainingResult

    # Create an algorithm object to train the implicit ALS model with the default method
    algorithm = training.Batch()

    # Pass a training data set and dependent values to the algorithm
    algorithm.input.setTable(training.data, dataTable)
    algorithm.input.setModel(training.inputModel, initialModel)

    algorithm.parameter.nFactors = nFactors

    # Build the implicit ALS model and retrieve the algorithm results
    trainingResult = algorithm.compute()


def testModel():

    # Create an algorithm object to predict recommendations of the implicit ALS model
    algorithm = prediction.ratings.Batch()
    algorithm.parameter.nFactors = nFactors

    algorithm.input.set(prediction.ratings.model, trainingResult.get(training.model))

    res = algorithm.compute()
    predictedRatings = res.get(prediction.ratings.prediction)

    printNumericTable(predictedRatings, "Predicted ratings:")

if __name__ == "__main__":

    initializeModel()
    trainModel()
    testModel()
