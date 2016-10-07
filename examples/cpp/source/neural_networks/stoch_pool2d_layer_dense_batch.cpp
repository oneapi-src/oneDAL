/* file: stoch_pool2d_layer_dense_batch.cpp */
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
!    C++ example of neural network forward and backward two-dimensional stochastic pooling layers usage.
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-STOCHASTIC_POOLING2D_LAYER_BATCH"></a>
 * \example stoch_pool2d_layer_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::data_management;
using namespace daal::services;

/* Input non-negative data set */
string datasetFileName = "../data/batch/layer_non_negative.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Read datasetFileName from a file and create a tensor to store input data */
    TensorPtr data  = readTensorFromCSV(datasetFileName);

    /* Get number of dimensions in input data */
    size_t nDim = data->getNumberOfDimensions();
    printTensor(data, "Forward two-dimensional stochastic pooling layer input (first 10 rows):", 10);

    /* Create an algorithm to compute forward two-dimensional stochastic pooling layer results using default method */
    stochastic_pooling2d::forward::Batch<> forwardLayer(nDim);
    forwardLayer.input.set(forward::data, data);

    /* Compute forward two-dimensional stochastic pooling layer results */
    forwardLayer.compute();

    /* Get the computed forward two-dimensional stochastic pooling layer results */
    services::SharedPtr<stochastic_pooling2d::forward::Result> forwardResult = forwardLayer.getResult();

    printTensor(forwardResult->get(forward::value), "Forward two-dimensional stochastic pooling layer result (first 5 rows):", 5);
    printTensor(forwardResult->get(stochastic_pooling2d::auxSelectedIndices),
                "Forward two-dimensional stochastic pooling layer selected indices (first 10 rows):", 10);

    /* Create an algorithm to compute backward two-dimensional stochastic pooling layer results using default method */
    stochastic_pooling2d::backward::Batch<> backwardLayer(nDim);
    backwardLayer.input.set(backward::inputGradient, forwardResult->get(forward::value));
    backwardLayer.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward two-dimensional stochastic pooling layer results */
    backwardLayer.compute();

    /* Get the computed backward two-dimensional stochastic pooling layer results */
    services::SharedPtr<backward::Result> backwardResult = backwardLayer.getResult();

    printTensor(backwardResult->get(backward::gradient),
                "Backward two-dimensional stochastic pooling layer result (first 10 rows):", 10);

    return 0;
}
