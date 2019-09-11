/* file: tanh_layer_dense_batch.cpp */
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
!    C++ example of forward and backward hyperbolic tangent (tanh) layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-TANH_LAYER_BATCH"></a>
 * \example tanh_layer_dense_batch.cpp
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

    /* Create an algorithm to compute forward hyperbolic tangent layer results using default method */
    tanh::forward::Batch<> tanhLayerForward;

    /* Set input objects for the forward hyperbolic tangent layer */
    tanhLayerForward.input.set(forward::data, tensorData);

    /* Compute forward hyperbolic tangent layer results */
    tanhLayerForward.compute();

    /* Print the results of the forward hyperbolic tangent layer */
    forward::ResultPtr forwardResult = tanhLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward tanh layer result (first 5 rows):", 5);

    /* Get the size of forward hyperbolic tangent layer output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward hyperbolic tangent layer results using default method */
    tanh::backward::Batch<> tanhLayerBackward;

    /* Set input objects for the backward hyperbolic tangent layer */
    tanhLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    tanhLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward hyperbolic tangent layer results */
    tanhLayerBackward.compute();

    /* Print the results of the backward hyperbolic tangent layer */
    backward::ResultPtr backwardResult = tanhLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward tanh layer result (first 5 rows):", 5);

    return 0;
}
