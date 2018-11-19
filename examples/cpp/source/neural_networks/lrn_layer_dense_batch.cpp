/* file: lrn_layer_dense_batch.cpp */
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
!    C++ example of forward and backward local response normalization (lrn) layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LRN_LAYER_BATCH"></a>
 * \example lrn_layer_dense_batch.cpp
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

    /* Create an algorithm to compute forward local response normalization layer results using default method */
    lrn::forward::Batch<> forwardLRNlayer;

    /* Set input objects for the forward local response normalization layer */
    forwardLRNlayer.input.set(forward::data, tensorData);

    /* Compute forward local response normalization layer results */
    forwardLRNlayer.compute();

    /* Print the results of the forward local response normalization layer */
    lrn::forward::ResultPtr forwardResult = forwardLRNlayer.getResult();
    printTensor(tensorData, "LRN layer input (first 5 rows):", 5);
    printTensor(forwardResult->get(forward::value), "LRN layer result (first 5 rows):", 5);
    printTensor(forwardResult->get(lrn::auxSmBeta), "LRN layer auxSmBeta (first 5 rows):", 5);

    /* Get the size of forward local response normalization layer output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward local response normalization layer results using default method */
    lrn::backward::Batch<> backwardLRNlayer;

    /* Set input objects for the backward local response normalization layer */
    backwardLRNlayer.input.set(backward::inputGradient, tensorDataBack);
    backwardLRNlayer.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward local response normalization layer results */
    backwardLRNlayer.compute();

    /* Print the results of the backward local response normalization layer */
    backward::ResultPtr backwardResult = backwardLRNlayer.getResult();
    printTensor(backwardResult->get(backward::gradient), "LRN layer backpropagation result (first 5 rows):", 5);

    return 0;
}
