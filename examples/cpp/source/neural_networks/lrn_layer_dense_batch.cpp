/* file: lrn_layer_dense_batch.cpp */
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
    services::SharedPtr<lrn::forward::Result> forwardResult = forwardLRNlayer.getResult();
    printTensor(tensorData, "LRN layer input (first 5 rows):", 5);
    printTensor(forwardResult->get(forward::value), "LRN layer result (first 5 rows):", 5);
    printTensor(forwardResult->get(lrn::auxSmBeta), "LRN layer auxSmBeta (first 5 rows):", 5);

    /* Get the size of forward local response normalization layer output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<float>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward local response normalization layer results using default method */
    lrn::backward::Batch<> backwardLRNlayer;

    /* Set input objects for the backward local response normalization layer */
    backwardLRNlayer.input.set(backward::inputGradient, tensorDataBack);
    backwardLRNlayer.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward local response normalization layer results */
    backwardLRNlayer.compute();

    /* Print the results of the backward local response normalization layer */
    services::SharedPtr<backward::Result> backwardResult = backwardLRNlayer.getResult();
    printTensor(backwardResult->get(backward::gradient), "LRN layer backpropagation result (first 5 rows):", 5);

    return 0;
}
