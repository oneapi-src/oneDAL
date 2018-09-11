/* file: batch_norm_layer_dense_batch.cpp */
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
!    C++ example of forward and backward batch normalization layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-BATCH_NORMALIZATION_LAYER_BATCH"></a>
 * \example batch_norm_layer_dense_batch.cpp
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
const size_t dimension = 1;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Read datasetFileName from a file and create a tensor to store input data */
    TensorPtr data  = readTensorFromCSV(datasetFileName);

    printTensor(data, "Forward batch normalization layer input (first 5 rows):", 5);

    /* Get collection of dimension sizes of the input data tensor */
    const Collection<size_t> &dataDims = data->getDimensions();
    size_t dimensionSize = dataDims[dimension];

    /* Create a collection of dimension sizes of input weights, biases, population mean and variance tensors */
    Collection<size_t> dimensionSizes;
    dimensionSizes.push_back(dimensionSize);

    /* Create input weights, biases, population mean and population variance tensors */
    TensorPtr weights(new HomogenTensor<>(dimensionSizes, Tensor::doAllocate, 1.0f));
    TensorPtr biases (new HomogenTensor<>(dimensionSizes, Tensor::doAllocate, 2.0f));
    TensorPtr populationMean    (new HomogenTensor<>(dimensionSizes, Tensor::doAllocate, 0.0f));
    TensorPtr populationVariance(new HomogenTensor<>(dimensionSizes, Tensor::doAllocate, 0.0f));

    /* Create an algorithm to compute forward batch normalization layer results using default method */
    batch_normalization::forward::Batch<> forwardLayer;
    forwardLayer.parameter.dimension = dimension;
    forwardLayer.input.set(forward::data,    data);
    forwardLayer.input.set(forward::weights, weights);
    forwardLayer.input.set(forward::biases,  biases);
    forwardLayer.input.set(batch_normalization::forward::populationMean,     populationMean);
    forwardLayer.input.set(batch_normalization::forward::populationVariance, populationVariance);

    /* Compute forward batch normalization layer results */
    forwardLayer.compute();

    /* Get the computed forward batch normalization layer results */
    batch_normalization::forward::ResultPtr forwardResult = forwardLayer.getResult();

    printTensor(forwardResult->get(forward::value), "Forward batch normalization layer result (first 5 rows):", 5);
    printTensor(forwardResult->get(batch_normalization::auxMean), "Mini-batch mean (first 5 values):", 5);
    printTensor(forwardResult->get(batch_normalization::auxStandardDeviation), "Mini-batch standard deviation (first 5 values):", 5);
    printTensor(forwardResult->get(batch_normalization::auxPopulationMean), "Population mean (first 5 values):", 5);
    printTensor(forwardResult->get(batch_normalization::auxPopulationVariance), "Population variance (first 5 values):", 5);

    /* Create input gradient tensor for backward batch normalization layer */
    TensorPtr inputGradientTensor = TensorPtr(new HomogenTensor<>(dataDims, Tensor::doAllocate, 10.0f));

    /* Create an algorithm to compute backward batch normalization layer results using default method */
    batch_normalization::backward::Batch<> backwardLayer;
    backwardLayer.parameter.dimension = dimension;
    backwardLayer.input.set(backward::inputGradient, inputGradientTensor);
    backwardLayer.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward batch normalization layer results */
    backwardLayer.compute();

    /* Get the computed backward batch normalization layer results */
    backward::ResultPtr backwardResult = backwardLayer.getResult();

    printTensor(backwardResult->get(backward::gradient), "Backward batch normalization layer result (first 5 rows):", 5);
    printTensor(backwardResult->get(backward::weightDerivatives), "Weight derivatives (first 5 values):", 5);
    printTensor(backwardResult->get(backward::biasDerivatives), "Bias derivatives (first 5 values):", 5);

    return 0;
}
