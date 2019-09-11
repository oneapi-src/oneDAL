/* file: elu_layer_dense_batch.cpp */
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
!    C++ example of forward and backward Exponential Linear Unit (ELU) layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-ELU_LAYER_BATCH"></a>
 * \example elu_layer_dense_batch.cpp
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

    /* Create an algorithm to compute forward ELU layer results using default method */
    elu::forward::Batch<> eluLayerForward;

    /* Set input objects for the forward ELU layer */
    eluLayerForward.input.set(forward::data, tensorData);

    /* Compute forward ELU layer results */
    eluLayerForward.compute();

    // /* Print the results of the forward ELU layer */
    elu::forward::ResultPtr forwardResult = eluLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward ELU layer result (first 5 rows):", 5);

    // /* Get the size of forward ELU layer output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<>(gDims, Tensor::doAllocate, 1.0f));

    // /* Create an algorithm to compute backward ELU layer results using default method */
    elu::backward::Batch<> eluLayerBackward;

    // /* Set input objects for the backward ELU layer */
    eluLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    eluLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    // /* Compute backward ELU layer results */
    eluLayerBackward.compute();

    // /* Print the results of the backward ELU layer */
    backward::ResultPtr backwardResult = eluLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward ELU layer result (first 5 rows):", 5);

    return 0;
}
