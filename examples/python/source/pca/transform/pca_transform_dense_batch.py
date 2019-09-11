# file: pca_transform_dense_batch.py
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
# !    Python example of PCA transformation algorithm.
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-PCA_TRANSFORM_DENSE_BATCH"></a>
## \example pca_transform_dense_batch.py
#

import os
import sys
import numpy as np

import daal.algorithms.pca as pca
import daal.algorithms.pca.transform as pca_transform
from daal.data_management import DataSourceIface, FileDataSource

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable
from daal.data_management import NumericTable
# Input data set parameters
datasetName = os.path.join('..', 'data', 'batch', 'pca_transform.csv')

if __name__ == "__main__":

    # Retrieve the input data
    dataSource = FileDataSource(datasetName,
                                DataSourceIface.doAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)
    dataSource.loadDataBlock()
    data = dataSource.getNumericTable()

    # Create an algorithm
    algorithm = pca.Batch(fptype=np.float64,method=pca.svdDense)

    # Set the algorithm input data
    algorithm.input.setDataset(pca.data, data)

    # Set the algorithm normalization parameters (mean and variance)
    # to be exported for transform and whitening parameter (eigenvalue)
    # If whitening is not required eigenvalues should be removed
    # The eigenvalues would be calculated in pca.eigenvalues table of result
    # but would not be passed to dataForTranform collection
    # algorithm.paramter.resultsToCompute = (pca.mean | pca.variance | pca.eigenvalue)

    algorithm.parameter.resultsToCompute = pca.mean | pca.variance | pca.eigenvalue;

    # Compute PCA
    res = algorithm.compute()
    # Output basis, eigenvalues and mean values
    printNumericTable(res.get(pca.eigenvalues), "Eigenvalues:")
    printNumericTable(res.get(pca.eigenvectors), "Eigenvectors:")

    eigenvaluesT = res.get(pca.eigenvalues)
    printNumericTable(eigenvaluesT, "Eigenvalues kv:")

    meansT = res.get(pca.means)
    printNumericTable(meansT, "Means kv:")

    #eigenvaluesT = res.getCollection(pca.eigenvalue)
    variancesT = res.get(pca.variances)
    printNumericTable(variancesT, "Variances kv:")

    # Create an algorithm
    tralgorithm = pca_transform.Batch(fptype=np.float64)

    # Set lower and upper bounds for the algorithm
    tralgorithm.parameter.nComponents = 2

    # Set an input object for the algorithm
    tralgorithm.input.setTable(pca_transform.data, data)

    # Set an input object for the eigenvectors
    tralgorithm.input.setTable(pca_transform.eigenvectors, res.get(pca.eigenvectors))

    # Set an input object for the eigenvectors
    tralgorithm.input.setCollection(pca_transform.dataForTransform, res.getCollection(pca.dataForTransform))

    # Compute PCA transformation function
    trres = tralgorithm.compute()

    printNumericTable(trres.get(pca.transform.transformedData), "Transformed data:");
    #printNumericTable(data, "First rows of the input data:", 4)
    #printNumericTable(trres.get(pca_transform.transformedData), "First rows of the min-max normalization result:", 4)
