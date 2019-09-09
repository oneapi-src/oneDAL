# file: adagrad_opt_res_dense_batch.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

#
# !  Content:
# !    Python example of the Adagrad algorithm
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-ADAGRAD_OPT_RES_DENSE_BATCH"></a>
##  \example adagrad_opt_res_dense_batch.py
#

import os
import sys

import numpy as np

import daal.algorithms.optimization_solver as optimization_solver
import daal.algorithms.optimization_solver.mse
import daal.algorithms.optimization_solver.adagrad
import daal.algorithms.optimization_solver.iterative_solver
from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable, MergedNumericTable, NumericTableIface
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

datasetFileName = os.path.join('..', 'data', 'batch', 'mse.csv')

nFeatures = 3
accuracyThreshold = 0.0000001
halfNIterations = 500
nIterations = halfNIterations * 2
batchSize = 1
learningRate = 1.0

startPoint = np.array([[8], [2], [1], [4]], dtype=np.float64)

if __name__ == "__main__":

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = FileDataSource(datasetFileName,
                                DataSourceIface.notAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)

    # Create Numeric Tables for data and values for dependent variable
    data = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    dependentVariables = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(data, dependentVariables)

    # Retrieve the data from the input file
    dataSource.loadDataBlock(mergedData)

    nVectors = data.getNumberOfRows()

    mseObjectiveFunction = optimization_solver.mse.Batch(nVectors)
    mseObjectiveFunction.input.set(optimization_solver.mse.data, data)
    mseObjectiveFunction.input.set(optimization_solver.mse.dependentVariables, dependentVariables)

    # Create objects to compute the Adagrad result using the default method
    adagradAlgorithm = optimization_solver.adagrad.Batch(mseObjectiveFunction)

    # Set input objects for the the Adagrad algorithm
    adagradAlgorithm.input.setInput(optimization_solver.iterative_solver.inputArgument, HomogenNumericTable(startPoint))
    adagradAlgorithm.parameter.learningRate = HomogenNumericTable(1, 1, NumericTableIface.doAllocate, learningRate)
    adagradAlgorithm.parameter.nIterations = halfNIterations
    adagradAlgorithm.parameter.accuracyThreshold = accuracyThreshold
    adagradAlgorithm.parameter.batchSize = batchSize
    adagradAlgorithm.parameter.optionalResultRequired = True

    # Compute the Adagrad result
    # Result class from daal.algorithms.optimization_solver.iterative_solver
    res = adagradAlgorithm.compute()

    # Print computed the Adagrad result
    printNumericTable(res.getResult(optimization_solver.iterative_solver.minimum), "Minimum after first compute():")
    printNumericTable(res.getResult(optimization_solver.iterative_solver.nIterations), "Number of iterations performed:")

    adagradAlgorithm.input.setInput(optimization_solver.iterative_solver.inputArgument, res.getResult(optimization_solver.iterative_solver.minimum))
    adagradAlgorithm.input.setInput(optimization_solver.iterative_solver.optionalArgument, res.getResult(optimization_solver.iterative_solver.optionalResult))

    res = adagradAlgorithm.compute()

    printNumericTable(res.getResult(optimization_solver.iterative_solver.minimum), "Minimum after second compute():")
    printNumericTable(res.getResult(optimization_solver.iterative_solver.nIterations), "Number of iterations performed:")
