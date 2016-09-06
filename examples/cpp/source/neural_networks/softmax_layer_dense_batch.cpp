/* file: softmax_layer_dense_batch.cpp */
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
!    C++ example of forward and backward softmax layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SOFTMAX_LAYER_BATCH"></a>
 * \example softmax_layer_dense_batch.cpp
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
const size_t dimension = 1;

int main()
{
    /* Read datasetFileName from a file and create a tensor to store input data */
    TensorPtr tensorData = readTensorFromCSV(datasetName);

    /* Create an algorithm to compute forward softmax layer results using default method */
    softmax::forward::Batch<> softmaxLayerForward;
    softmaxLayerForward.parameter.dimension = dimension;

    /* Set input objects for the forward softmax layer */
    softmaxLayerForward.input.set(forward::data, tensorData);

    /* Compute forward softmax layer results */
    softmaxLayerForward.compute();

    /* Print the results of the forward softmax layer */
    services::SharedPtr<softmax::forward::Result> forwardResult = softmaxLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward softmax layer result (first 5 rows):", 5);

    /* Get the size of forward softmax layer output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<float>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward softmax layer results using default method */
    softmax::backward::Batch<> softmaxLayerBackward;
    softmaxLayerBackward.parameter.dimension = dimension;

    /* Set input objects for the backward softmax layer */
    softmaxLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    softmaxLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward softmax layer results */
    softmaxLayerBackward.compute();

    /* Print the results of the backward softmax layer */
    services::SharedPtr<backward::Result> backwardResult = softmaxLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward softmax layer result (first 5 rows):", 5);

    return 0;
}
