# file: mse_dense_batch.py
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
# !    Python example of the mean squared error objective function
# !*****************************************************************************


#
## <a name="DAAL-EXAMPLE-PY-MSE_BATCH"></a>
##  \example mse_dense_batch.py
#

import os
import sys

import numpy as np

import daal.algorithms.optimization_solver as optimization_solver
import daal.algorithms.optimization_solver.mse

from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable, MergedNumericTable, NumericTableIface
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

datasetFileName = os.path.join('..', 'data', 'batch', 'mse.csv')
nFeatures = 3

argumentValue = np.array([[-1], [0.1], [0.15], [-0.5]], dtype=np.float64)

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

    # Create the MSE objective function objects to compute the MSE objective function result using the default method
    mseObjectiveFunction = optimization_solver.mse.Batch(nVectors)

    # Set input objects for the MSE objective function
    mseObjectiveFunction.input.set(optimization_solver.mse.data, data)
    mseObjectiveFunction.input.set(optimization_solver.mse.dependentVariables, dependentVariables)
    mseObjectiveFunction.input.set(optimization_solver.mse.argument, HomogenNumericTable(argumentValue))
    mseObjectiveFunction.parameter().resultsToCompute = (
        optimization_solver.objective_function.gradient |
        optimization_solver.objective_function.value |
        optimization_solver.objective_function.hessian
    )

    # Compute the MSE objective function result
    # Result class from optimization_solver.objective_function
    res = mseObjectiveFunction.compute()

    # Print computed the MSE objective function result
    printNumericTable(res.get(optimization_solver.objective_function.valueIdx),
                      "Value")
    printNumericTable(res.get(optimization_solver.objective_function.gradientIdx),
                      "Gradient")
    printNumericTable(res.get(optimization_solver.objective_function.hessianIdx),
                      "Hessian")
