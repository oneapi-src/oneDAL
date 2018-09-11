/* file: softmax_layer_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
!    C++ example of forward and backward softmax layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SOFTMAX_LAYER_BATCH"></a>
 * \example softmax_layer_dense_batch.cpp
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
const size_t dimension = 1;

int main()
{
    /* Read datasetFileName from a file and create a tensor to store input data */
    TensorPtr tensorData = readTensorFromCSV(datasetName);

    /* Create an algorithm to compute forward softmax layer results using default method */
    softmax::forward::Batch<> softmaxLayerForward;
    softmaxLayerForward.parameter.dimension = dimension;

    /* Set input objects for the forward softmax layer */
    softmaxLayerForward.input.set(forward::data, tensorData);

    /* Compute forward softmax layer results */
    softmaxLayerForward.compute();

    /* Print the results of the forward softmax layer */
    softmax::forward::ResultPtr forwardResult = softmaxLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward softmax layer result (first 5 rows):", 5);

    /* Get the size of forward softmax layer output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward softmax layer results using default method */
    softmax::backward::Batch<> softmaxLayerBackward;
    softmaxLayerBackward.parameter.dimension = dimension;

    /* Set input objects for the backward softmax layer */
    softmaxLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    softmaxLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward softmax layer results */
    softmaxLayerBackward.compute();

    /* Print the results of the backward softmax layer */
    backward::ResultPtr backwardResult = softmaxLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward softmax layer result (first 5 rows):", 5);

    return 0;
}
