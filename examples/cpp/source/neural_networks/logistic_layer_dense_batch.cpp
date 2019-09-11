/* file: logistic_layer_dense_batch.cpp */
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
!    C++ example of forward and backward logistic layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LOGISTIC_LAYER_BATCH"></a>
 * \example logistic_layer_dense_batch.cpp
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
    /* Retrieve the input data */
    TensorPtr tensorData = readTensorFromCSV(datasetName);

    /* Create an algorithm to compute forward logistic layer results using default method */
    logistic::forward::Batch<> logisticLayerForward;

    /* Set input objects for the forward logistic layer */
    logisticLayerForward.input.set(forward::data, tensorData);

    /* Compute forward logistic layer results */
    logisticLayerForward.compute();

    /* Print the results of the forward logistic layer */
    logistic::forward::ResultPtr forwardResult = logisticLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward logistic layer result (first 5 rows):", 5);

    /* Get the size of forward logistic layer output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward logistic layer results using default method */
    logistic::backward::Batch<> logisticLayerBackward;

    /* Set input objects for the backward logistic layer */
    logisticLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    logisticLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward logistic layer results */
    logisticLayerBackward.compute();

    /* Print the results of the backward logistic layer */
    backward::ResultPtr backwardResult = logisticLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward logistic layer result (first 5 rows):", 5);

    return 0;
}
