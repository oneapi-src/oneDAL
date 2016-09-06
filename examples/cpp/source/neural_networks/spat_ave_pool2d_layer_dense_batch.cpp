/* file: spat_ave_pool2d_layer_dense_batch.cpp */
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
!    C++ example of neural network forward and backward two-dimensional average pooling layers usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SPAT_AVE_POOL2D_LAYER_DENSE_BATCH"></a>
 * \example spat_ave_pool2d_layer_dense_batch.cpp
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
static float dataArray[2][3][2][4] =
                                    {{{{2, 4, 6, 8},
                                       {10, 12, 14, 16}},
                                                      {{18, 20, 22, 24},
                                                       {26, 28, 30, 32}},
                                                                      {{34, 36, 38, 40},
                                                                       {42, 44, 46, 48}}},
                                    {{{ -2, -4, -6, -8},
                                      { -10, -12, -14, -16}},
                                                          {{ -18, -20, -22, -24},
                                                           { -26, -28, -30, -32}},
                                                                               {{ -34, -36, -38, -40},
                                                                                { -42, -44, -46, -48}}}};

int main(int argc, char *argv[])
{
    TensorPtr data(new HomogenTensor<float>(nDim, dims, (float *)dataArray));
    printTensor(data, "Forward two-dimensional spatial pyramid average pooling layer input (first 10 rows):", 10);

    /* Create an algorithm to compute forward two-dimensional spatial pyramid average pooling layer results using default method */
    spatial_average_pooling2d::forward::Batch<> forwardLayer(2, nDim);
    forwardLayer.input.set(forward::data, data);

    /* Compute forward two-dimensional spatial pyramid average pooling layer results */
    forwardLayer.compute();

    /* Get the computed forward two-dimensional spatial pyramid average pooling layer results */
    services::SharedPtr<spatial_average_pooling2d::forward::Result> forwardResult = forwardLayer.getResult();

    printTensor(forwardResult->get(forward::value), "Forward two-dimensional spatial pyramid average pooling layer result (first 5 rows):", 5);
    printNumericTable(forwardResult->get(spatial_average_pooling2d::auxInputDimensions), "Forward two-dimensional spatial pyramid average pooling layer input dimensions:");

    /* Create an algorithm to compute backward two-dimensional spatial pyramid average pooling layer results using default method */
    spatial_average_pooling2d::backward::Batch<> backwardLayer(2, nDim);
    backwardLayer.input.set(backward::inputGradient, forwardResult->get(forward::value));
    backwardLayer.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward two-dimensional spatial pyramid average pooling layer results */
    backwardLayer.compute();

    /* Get the computed backward two-dimensional spatial pyramid average pooling layer results */
    services::SharedPtr<backward::Result> backwardResult = backwardLayer.getResult();

    printTensor(backwardResult->get(backward::gradient),
        "Backward two-dimensional spatial pyramid average pooling layer result (first 10 rows):", 10);

    return 0;
}
