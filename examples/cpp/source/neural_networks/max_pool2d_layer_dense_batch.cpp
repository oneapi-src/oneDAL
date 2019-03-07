/* file: max_pool2d_layer_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
!  Content:
!    C++ example of neural network forward and backward two-dimensional maximum pooling layers usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-MAXIMUM_POOLING2D_LAYER_BATCH"></a>
 * \example max_pool2d_layer_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::data_management;
using namespace daal::services;

/* Input data set name */
string datasetFileName = "../data/batch/layer.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Read datasetFileName from a file and create a tensor to store input data */
    TensorPtr data  = readTensorFromCSV(datasetFileName);
    size_t nDim = data->getNumberOfDimensions();

    printTensor(data, "Forward two-dimensional maximum pooling layer input (first 10 rows):", 10);

    /* Create an algorithm to compute forward two-dimensional maximum pooling layer results using default method */
    maximum_pooling2d::forward::Batch<> forwardLayer(nDim);
    forwardLayer.input.set(forward::data, data);

    /* Compute forward two-dimensional maximum pooling layer results */
    forwardLayer.compute();

    /* Get the computed forward two-dimensional maximum pooling layer results */
    maximum_pooling2d::forward::ResultPtr forwardResult = forwardLayer.getResult();

    printTensor(forwardResult->get(forward::value), "Forward two-dimensional maximum pooling layer result (first 5 rows):", 5);
    printTensor(forwardResult->get(maximum_pooling2d::auxSelectedIndices),
        "Forward two-dimensional maximum pooling layer selected indices (first 10 rows):", 10);

    /* Create an algorithm to compute backward two-dimensional maximum pooling layer results using default method */
    maximum_pooling2d::backward::Batch<> backwardLayer(nDim);
    backwardLayer.input.set(backward::inputGradient, forwardResult->get(forward::value));
    backwardLayer.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward two-dimensional maximum pooling layer results */
    backwardLayer.compute();

    /* Get the computed backward two-dimensional maximum pooling layer results */
    backward::ResultPtr backwardResult = backwardLayer.getResult();

    printTensor(backwardResult->get(backward::gradient),
        "Backward two-dimensional maximum pooling layer result (first 10 rows):", 10);

    return 0;
}
