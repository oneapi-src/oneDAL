/* file: dropout_layer_dense_batch.cpp */
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
!    C++ example of forward and backward dropout layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DROPOUT_LAYER_BATCH"></a>
 * \example dropout_layer_dense_batch.cpp
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

    /* Create an algorithm to compute forward dropout layer results using default method */
    dropout::forward::Batch<> dropoutLayerForward;

    /* Set input objects for the forward dropout layer */
    dropoutLayerForward.input.set(forward::data, tensorData);

    /* Compute forward dropout layer results */
    dropoutLayerForward.compute();

    /* Print the results of the forward dropout layer */
    dropout::forward::ResultPtr forwardResult = dropoutLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward dropout layer result (first 5 rows):", 5);
    printTensor(forwardResult->get(dropout::auxRetainMask), "Dropout layer retain mask (first 5 rows):", 5);

    /* Get the size of forward dropout layer output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward dropout layer results using default method */
    dropout::backward::Batch<> dropoutLayerBackward;

    /* Set input objects for the backward dropout layer */
    dropoutLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    dropoutLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward dropout layer results */
    dropoutLayerBackward.compute();

    /* Print the results of the backward dropout layer */
    backward::ResultPtr backwardResult = dropoutLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward dropout layer result (first 5 rows):", 5);

    return 0;
}
