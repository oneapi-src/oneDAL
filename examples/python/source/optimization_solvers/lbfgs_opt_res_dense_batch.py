# file: lbfgs_opt_res_dense_batch.py
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

#
# !  Content:
# !    Python example of the LBFGS algorithm
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-LBFGS_OPT_RES_DENSE_BATCH"></a>
##  \example lbfgs_opt_res_dense_batch.py
#

import os
import sys

import numpy as np

import daal.algorithms.optimization_solver as optimization_solver
import daal.algorithms.optimization_solver.mse
import daal.algorithms.optimization_solver.lbfgs
import daal.algorithms.optimization_solver.iterative_solver

from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable, MergedNumericTable, NumericTableIface
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

datasetFileName = os.path.join('..', 'data', 'batch', 'lbfgs.csv')

nFeatures = 10
halfNIterations = 500
nIterations = halfNIterations * 2
stepLength = 1.0e-4

initialPoint = np.array([[100], [100], [100], [100], [100], [100], [100], [100], [100], [100], [100]], dtype=np.float64)
expectedPoint = np.array([[11], [  1], [  2], [  3], [  4], [  5], [  6], [  7], [  8], [  9], [ 10]], dtype=np.float64)

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

    mseObjectiveFunction = optimization_solver.mse.Batch(data.getNumberOfRows())
    mseObjectiveFunction.input.set(optimization_solver.mse.data, data)
    mseObjectiveFunction.input.set(optimization_solver.mse.dependentVariables, dependentVariables)

    # Create objects to compute the lbfgs result using the default method
    lbfgsAlgorithm = optimization_solver.lbfgs.Batch(mseObjectiveFunction)
    lbfgsAlgorithm.parameter.nIterations = halfNIterations
    lbfgsAlgorithm.parameter.stepLengthSequence = HomogenNumericTable(1, 1, NumericTableIface.doAllocate, stepLength)
    lbfgsAlgorithm.parameter.optionalResultRequired = True

    # Set input objects for LBFGS algorithm
    lbfgsAlgorithm.input.setInput(optimization_solver.iterative_solver.inputArgument, HomogenNumericTable(initialPoint))

    # Compute the lbfgs result
    # Result class from daal.algorithms.optimization_solver.iterative_solver
    res = lbfgsAlgorithm.compute()

    # Print computed the lbfgs result
    printNumericTable(res.getResult(optimization_solver.iterative_solver.minimum), "Resulting coefficients after first compute():")
    printNumericTable(res.getResult(optimization_solver.iterative_solver.nIterations), "Number of iterations performed:")

    # Set optional result as an optional input
    lbfgsAlgorithm.input.setInput(optimization_solver.iterative_solver.inputArgument, res.getResult(optimization_solver.iterative_solver.minimum))
    lbfgsAlgorithm.input.setInput(optimization_solver.iterative_solver.optionalArgument, res.getResult(optimization_solver.iterative_solver.optionalResult))

    # Print computed the lbfgs result
    res = lbfgsAlgorithm.compute()

    expectedCoefficients = HomogenNumericTable(expectedPoint)

    # Print computed the Adaptive gradient descent result
    printNumericTable(expectedCoefficients, "Expected coefficients:")
    printNumericTable(res.getResult(optimization_solver.iterative_solver.minimum), "Resulting coefficients after second compute():")
    printNumericTable(res.getResult(optimization_solver.iterative_solver.nIterations), "Number of iterations performed:")
