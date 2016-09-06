/* file: max_pool1d_layer_dense_batch.cpp */
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
!    C++ example of neural network forward and backward one-dimensional maximum pooling layers usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-MAXIMUM_POOLING1D_LAYER_BATCH"></a>
 * \example max_pool1d_layer_dense_batch.cpp
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

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Read datasetFileName from a file and create a tensor to store input data */
    TensorPtr data  = readTensorFromCSV(datasetFileName);
    size_t nDim = data->getNumberOfDimensions();

    printTensor(data, "Forward one-dimensional maximum pooling layer input (first 10 rows):", 10);

    /* Create an algorithm to compute forward one-dimensional pooling layer results using maximum method */
    maximum_pooling1d::forward::Batch<> forwardLayer(nDim);
    forwardLayer.input.set(forward::data, data);

    /* Compute forward one-dimensional maximum pooling layer results */
    forwardLayer.compute();

    /* Print the results of the forward one-dimensional maximum pooling layer */
    services::SharedPtr<maximum_pooling1d::forward::Result> forwardResult = forwardLayer.getResult();

    printTensor(forwardResult->get(forward::value), "Forward one-dimensional maximum pooling layer result (first 5 rows):", 5);
    printTensor(forwardResult->get(maximum_pooling1d::auxSelectedIndices),
        "Forward one-dimensional maximum pooling layer selected indices (first 5 rows):", 5);

    /* Create an algorithm to compute backward one-dimensional maximum pooling layer results using default method */
    maximum_pooling1d::backward::Batch<> backwardLayer(nDim);

    /* Set input objects for the backward one-dimensional maximum pooling layer */
    backwardLayer.input.set(backward::inputGradient, forwardResult->get(forward::value));
    backwardLayer.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward one-dimensional maximum pooling layer results */
    backwardLayer.compute();

    /* Print the results of the backward one-dimensional maximum pooling layer */
    services::SharedPtr<backward::Result> backwardResult = backwardLayer.getResult();

    printTensor(backwardResult->get(backward::gradient),
                "Backward one-dimensional maximum pooling layer result (first 10 rows):", 10);

    return 0;
}
