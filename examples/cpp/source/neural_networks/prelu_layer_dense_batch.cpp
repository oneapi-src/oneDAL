/* file: prelu_layer_dense_batch.cpp */
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
!    C++ example of forward and backward parametric rectified linear unit (prelu) layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PRELU_LAYER_BATCH"></a>
 * \example prelu_layer_dense_batch.cpp
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
string weightsName = "../data/batch/layer.csv";

size_t dataDimension = 0;
size_t weightsDimension = 2;

int main()
{
    /* Read datasetFileName from a file and create a tensor to store input data */
    TensorPtr tensorData = readTensorFromCSV(datasetName);
    TensorPtr tensorWeights = readTensorFromCSV(weightsName);

    /* Create an algorithm to compute forward prelu layer results using default method */
    prelu::forward::Batch<> forwardPreluLayer;
    forwardPreluLayer.parameter.dataDimension = dataDimension;
    forwardPreluLayer.parameter.weightsDimension = weightsDimension;
    forwardPreluLayer.parameter.weightsAndBiasesInitialized = true;

    /* Set input objects for the forward prelu layer */
    forwardPreluLayer.input.set(forward::data, tensorData);
    forwardPreluLayer.input.set(forward::weights, tensorWeights);

    /* Compute forward prelu layer results */
    forwardPreluLayer.compute();

    /* Print the results of the forward prelu layer */
    prelu::forward::ResultPtr forwardResult = forwardPreluLayer.getResult();
    printTensor(forwardResult->get(forward::value), "Forward prelu layer result (first 5 rows):", 5);

    /* Get the size of forward prelu layer output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward prelu layer results using default method */
    prelu::backward::Batch<> backwardPreluLayer;
    backwardPreluLayer.parameter.dataDimension = dataDimension;
    backwardPreluLayer.parameter.weightsDimension = weightsDimension;

    /* Set input objects for the backward prelu layer */
    backwardPreluLayer.input.set(backward::inputGradient, tensorDataBack);
    backwardPreluLayer.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward prelu layer results */
    backwardPreluLayer.compute();

    /* Print the results of the backward prelu layer */
    backward::ResultPtr backwardResult = backwardPreluLayer.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward prelu layer result (first 5 rows):", 5);
    printTensor(backwardResult->get(backward::weightDerivatives), "Weights derivative (first 5 rows):", 5);

    return 0;
}
