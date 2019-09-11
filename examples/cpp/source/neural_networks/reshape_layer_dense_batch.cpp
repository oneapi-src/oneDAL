/* file: reshape_layer_dense_batch.cpp */
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
!    C++ example of forward and backward reshap layer usage
!
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-RESHAPE_LAYER_BATCH"></a>
 * \example reshape_layer_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::data_management;
using namespace daal::services;

/* Input data set parameters */
string datasetName = "../data/batch/layer.csv";

int main()
{
    /* Read datasetFileName from a file and create a tensor to store input data */
    TensorPtr tensorData = readTensorFromCSV(datasetName);

    /* Create an algorithm to compute forward reshape layer results using default method */
    reshape::forward::Batch<> reshapeLayerForward;

    /* Set input objects for the forward reshape layer */
    reshapeLayerForward.input.set(forward::data, tensorData);

    reshapeLayerForward.parameter.reshapeDimensions.push_back(-1).push_back(5);

    /* Compute forward reshape layer results */
    reshapeLayerForward.compute();

    /* Print the input of the forward reshape layer */
    printTensor(tensorData, "Forward reshape layer input (first 5 rows):", 5);

    /* Print the results of the forward reshape layer */
    reshape::forward::ResultPtr forwardResult = reshapeLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward reshape layer result (first 5 rows):", 5);

    /* Get the size of forward reshape layer output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward reshape layer results using default method */
    reshape::backward::Batch<> reshapeLayerBackward;

    /* Set input objects for the backward reshape layer */
    reshapeLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    reshapeLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward reshape layer results */
    reshapeLayerBackward.compute();

    /* Print the results of the backward reshape layer */
    backward::ResultPtr backwardResult = reshapeLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward reshape layer result (first 5 rows):", 5);

    return 0;
}
