/* file: relu_layer_dense_batch.cpp */
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
!    C++ example of forward and backward rectified linear unit (relu) layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-RELU_LAYER_BATCH"></a>
 * \example relu_layer_dense_batch.cpp
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

    /* Create an algorithm to compute forward relu layer results using default method */
    relu::forward::Batch<> reluLayerForward;

    /* Set input objects for the forward relu layer */
    reluLayerForward.input.set(forward::data, tensorData);

    /* Compute forward relu layer results */
    reluLayerForward.compute();

    /* Print the results of the forward relu layer */
    relu::forward::ResultPtr forwardResult = reluLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward relu layer result (first 5 rows):", 5);

    /* Get the size of forward relu layer output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward relu layer results using default method */
    relu::backward::Batch<> reluLayerBackward;

    /* Set input objects for the backward relu layer */
    reluLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    reluLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward relu layer results */
    reluLayerBackward.compute();

    /* Print the results of the backward relu layer */
    backward::ResultPtr backwardResult = reluLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward relu layer result (first 5 rows):", 5);

    return 0;
}
