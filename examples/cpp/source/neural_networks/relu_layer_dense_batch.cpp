/* file: relu_layer_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
    services::SharedPtr<relu::forward::Result> forwardResult = reluLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward relu layer result (first 5 rows):", 5);

    /* Get the size of forward relu layer output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<float>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward relu layer results using default method */
    relu::backward::Batch<> reluLayerBackward;

    /* Set input objects for the backward relu layer */
    reluLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    reluLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward relu layer results */
    reluLayerBackward.compute();

    /* Print the results of the backward relu layer */
    services::SharedPtr<backward::Result> backwardResult = reluLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward relu layer result (first 5 rows):", 5);

    return 0;
}
