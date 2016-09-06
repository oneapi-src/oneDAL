/* file: spat_stoch_pool2d_layer_dense_batch.cpp */
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
!    C++ example of neural network forward and backward two-dimensional spatial pyramid stochastic pooling layers usage.
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SPAT_STOCH_POOL2D_LAYER_DENSE_BATCH"></a>
 * \example spat_stoch_pool2d_layer_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::algorithms::neural_networks;
using namespace daal::data_management;
using namespace daal::services;

static const size_t nDim = 4;
static const size_t dims[] = {2, 3, 2, 4};
static float dataArray[2][3][2][4] = {{{{ 1,  2,  3,  4},
                                    { 5,  6,  7,  8}},
                                                    {{ 9, 10, 11, 12},
                                                    {13, 14, 15, 16}},
                                                                    {{17, 18, 19, 20},
                                                                     {21, 22, 23, 24}}},
                                  {{{ 10, 20, 30, 40},
                                    { 50, 60, 70, 80}},
                                                    {{ 90, 100, 110, 120},
                                                    {130, 140, 150, 160}},
                                                                    {{170, 180, 190, 200},
                                                                     {210, 220, 230, 240}}}};

int main(int argc, char *argv[])
{
    TensorPtr data(new HomogenTensor<float>(nDim, dims, (float *)dataArray));

    printTensor(data, "Forward two-dimensional spatial pyramid stochastic pooling layer input (first 10 rows):", 10);

    /* Create an algorithm to compute forward two-dimensional spatial pyramid stochastic pooling layer results using default method */
    spatial_stochastic_pooling2d::forward::Batch<> forwardLayer(2, nDim);
    forwardLayer.input.set(forward::data, data);

    /* Compute forward two-dimensional spatial pyramid stochastic pooling layer results */
    forwardLayer.compute();

    /* Get the computed forward two-dimensional spatial pyramid stochastic pooling layer results */
    services::SharedPtr<spatial_stochastic_pooling2d::forward::Result> forwardResult = forwardLayer.getResult();

    printTensor(forwardResult->get(forward::value), "Forward two-dimensional spatial pyramid stochastic pooling layer result (first 5 rows):", 5);
    printTensor(forwardResult->get(spatial_stochastic_pooling2d::auxSelectedIndices),
                "Forward two-dimensional spatial pyramid stochastic pooling layer selected indices (first 10 rows):", 10);

    /* Create an algorithm to compute backward two-dimensional spatial pyramid stochastic pooling layer results using default method */
    spatial_stochastic_pooling2d::backward::Batch<> backwardLayer(2, nDim);
    backwardLayer.input.set(backward::inputGradient, forwardResult->get(forward::value));
    backwardLayer.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward two-dimensional spatial pyramid stochastic pooling layer results */
    backwardLayer.compute();

    /* Get the computed backward two-dimensional spatial pyramid stochastic pooling layer results */
    services::SharedPtr<backward::Result> backwardResult = backwardLayer.getResult();

    printTensor(backwardResult->get(backward::gradient),
                "Backward two-dimensional spatial pyramid stochastic pooling layer result (first 10 rows):", 10);

    return 0;
}
