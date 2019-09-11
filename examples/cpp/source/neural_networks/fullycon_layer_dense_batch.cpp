/* file: fullycon_layer_dense_batch.cpp */
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
!    C++ example of forward and backward fully-connected layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-FULLYCONNECTED_LAYER_BATCH"></a>
 * \example fullycon_layer_dense_batch.cpp
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
    size_t m = 5;
    /* Read datasetFileName from a file and create a tensor to store input data */
    TensorPtr tensorData = readTensorFromCSV(datasetName);

    /* Create an algorithm to compute forward fully-connected layer results using default method */
    fullyconnected::forward::Batch<> fullyconnectedLayerForward(m);

    /* Set input objects for the forward fully-connected layer */
    fullyconnectedLayerForward.input.set(forward::data, tensorData);

    /* Compute forward fully-connected layer results */
    fullyconnectedLayerForward.compute();

    /* Print the results of the forward fully-connected layer */
    fullyconnected::forward::ResultPtr forwardResult = fullyconnectedLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward fully-connected layer result (first 5 rows):", 5);
    printTensor(forwardResult->get(fullyconnected::auxWeights), "Forward fully-connected layer weights (first 5 rows):", 5);

    /* Get the size of forward fully-connected layer output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward fully-connected layer results using default method */
    fullyconnected::backward::Batch<> fullyconnectedLayerBackward(m);

    /* Set input objects for the backward fully-connected layer */
    fullyconnectedLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    fullyconnectedLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward fully-connected layer results */
    fullyconnectedLayerBackward.compute();

    /* Print the results of the backward fully-connected layer */
    backward::ResultPtr backwardResult = fullyconnectedLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient),
                "Backward fully-connected layer gradient result (first 5 rows):", 5);
    printTensor(backwardResult->get(backward::weightDerivatives),
                "Backward fully-connected layer weightDerivative result (first 5 rows):", 5);
    printTensor(backwardResult->get(backward::biasDerivatives),
                "Backward fully-connected layer biasDerivative result (first 5 rows):", 5);

    return 0;
}
