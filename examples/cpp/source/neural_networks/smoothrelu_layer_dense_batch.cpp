/* file: smoothrelu_layer_dense_batch.cpp */
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
!    C++ example of forward and backward smooth rectified linear unit (smooth relu) layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SMOOTHRELU_LAYER_BATCH"></a>
 * \example smoothrelu_layer_dense_batch.cpp
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

    /* Create an algorithm to compute forward smooth relu layer results using default method */
    smoothrelu::forward::Batch<> smoothreluLayerForward;

    /* Set input objects for the forward smooth relu layer */
    smoothreluLayerForward.input.set(forward::data, tensorData);

    /* Compute forward smooth relu layer results */
    smoothreluLayerForward.compute();

    /* Print the results of the forward smooth relu layer */
    smoothrelu::forward::ResultPtr forwardResult = smoothreluLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward smooth ReLU layer result (first 5 rows):", 5);

    /* Get the size of forward dropout smooth relu output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward smooth relu layer results using default method */
    smoothrelu::backward::Batch<> smoothreluLayerBackward;

    /* Set input objects for the backward smooth relu layer */
    smoothreluLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    smoothreluLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward smooth relu layer results */
    smoothreluLayerBackward.compute();

    /* Print the results of the backward smooth relu layer */
    backward::ResultPtr backwardResult = smoothreluLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward smooth ReLU layer result (first 5 rows):", 5);

    return 0;
}
