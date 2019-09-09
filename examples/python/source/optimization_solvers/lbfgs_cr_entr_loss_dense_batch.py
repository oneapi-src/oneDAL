# file: lbfgs_cr_entr_loss_dense_batch.py
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

# !  Content:
# !    Python example of the limited memory Broyden-Fletcher-Goldfarb-Shanno
# !    algorithm with cross entropy loss function
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-LBFGS_CR_ENTR_LOSS_BATCH"></a>
##  \example lbfgs_cr_entr_loss_dense_batch.py
#

import os
import sys

import numpy as np

import daal.algorithms.optimization_solver as optimization_solver
import daal.algorithms.optimization_solver.cross_entropy_loss
import daal.algorithms.optimization_solver.lbfgs
import daal.algorithms.optimization_solver.iterative_solver

from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable, MergedNumericTable, NumericTableIface
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

datasetFileName = os.path.join('..', 'data', 'batch', 'logreg_train.csv')

nFeatures = 6
nClasses = 5
nIterations = 1000
stepLength = 1.0e-4

if __name__ == "__main__":

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = FileDataSource(datasetFileName,
                                DataSourceIface.notAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)

    # Create Numeric Tables for input data and dependent variables
    data = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    dependentVariables = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(data, dependentVariables)

    # Retrieve the data from input file
    dataSource.loadDataBlock(mergedData)

    func = optimization_solver.cross_entropy_loss.Batch(nClasses, data.getNumberOfRows())
    func.input.set(optimization_solver.cross_entropy_loss.data, data)
    func.input.set(optimization_solver.cross_entropy_loss.dependentVariables, dependentVariables)

    # Create objects to compute LBFGS result using the default method
    algorithm = optimization_solver.lbfgs.Batch(func)
    algorithm.parameter.nIterations = nIterations
    algorithm.parameter.stepLengthSequence = HomogenNumericTable(1, 1, NumericTableIface.doAllocate, stepLength)

    # Set input objects for LBFGS algorithm
    nParameters = nClasses * (nFeatures + 1)
    initialPoint = np.full((nParameters, 1), 0.001, dtype=np.float64)
    algorithm.input.setInput(optimization_solver.iterative_solver.inputArgument, HomogenNumericTable(initialPoint))

    # Compute LBFGS result
    # Result class from daal.algorithms.optimization_solver.iterative_solver
    res = algorithm.compute()

    expectedPoint = np.array([[-2.277], [2.836], [14.985], [0.511], [7.510], [-2.831], [-5.814], [-0.033], [13.227], [-24.447], [3.730],
        [10.394], [-10.461], [-0.766], [0.077], [1.558], [-1.133], [2.884], [-3.825], [7.699], [2.421], [-0.135], [-6.996], [1.785], [-2.294], [-9.819], [1.692],
        [-0.725], [0.069], [-8.41], [1.458], [-3.306], [-4.719], [5.507], [-1.642]], dtype=np.float64)
    expectedCoefficients = HomogenNumericTable(expectedPoint)

    # Print computed LBFGS results
    printNumericTable(expectedCoefficients, "Expected coefficients:")
    printNumericTable(res.getResult(optimization_solver.iterative_solver.minimum), "Resulting coefficients:")
    printNumericTable(res.getResult(optimization_solver.iterative_solver.nIterations), "Number of iterations performed:")
