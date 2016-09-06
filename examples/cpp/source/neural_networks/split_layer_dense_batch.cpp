/* file: split_layer_dense_batch.cpp */
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
!    C++ example of forward and backward split layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SPLIT_LAYER_BATCH"></a>
 * \example split_layer_dense_batch.cpp
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
const size_t nOutputs = 3;
const size_t nInputs  = 3;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetName);

    /* Read datasetFileName from a file and create a tensor to store input data */
    TensorPtr tensorData = readTensorFromCSV(datasetName);

    /* Create an algorithm to compute forward split layer results using default method */
    split::forward::Batch<> splitLayerForward;

    /* Set parameters for the forward split layer */
    splitLayerForward.parameter.nOutputs = nOutputs;
    splitLayerForward.parameter.nInputs = nInputs;

    /* Set input objects for the forward split layer */
    splitLayerForward.input.set(forward::data, tensorData);

    printTensor(tensorData, "Split layer input (first 5 rows):", 5);

    /* Compute forward split layer results */
    splitLayerForward.compute();

    /* Print the results of the forward split layer */
    services::SharedPtr<split::forward::Result> forwardResult = splitLayerForward.getResult();

    for(size_t i = 0; i < nOutputs; i++)
    {
        printTensor(forwardResult->get(split::forward::valueCollection, i), "Forward split layer result (first 5 rows):", 5);
    }

    /* Create an algorithm to compute backward split layer results using default method */
    split::backward::Batch<> splitLayerBackward;

    /* Set parameters for the backward split layer */
    splitLayerBackward.parameter.nOutputs = nOutputs;
    splitLayerBackward.parameter.nInputs = nInputs;

    /* Set input objects for the backward split layer */
    splitLayerBackward.input.set(split::backward::inputGradientCollection, forwardResult->get(split::forward::valueCollection));

    /* Compute backward split layer results */
    splitLayerBackward.compute();

    /* Print the results of the backward split layer */
    services::SharedPtr<backward::Result> backwardResult = splitLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward split layer result (first 5 rows):", 5);

    return 0;
}
