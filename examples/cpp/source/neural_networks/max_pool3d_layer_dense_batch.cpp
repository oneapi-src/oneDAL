/* file: max_pool3d_layer_dense_batch.cpp */
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
!    C++ example of neural network forward and backward three-dimensional maximum pooling layers usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-MAXIMUM_POOLING3D_LAYER_BATCH"></a>
 * \example max_pool3d_layer_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::data_management;
using namespace daal::services;

static const size_t nDim = 3;
static const size_t dims[] = {3, 2, 4};
static float dataArray[3][2][4] = {{{ 1,  2,  3,  4},
                                    { 5,  6,  7,  8}},
                                                    {{ 9, 10, 11, 12},
                                                    {13, 14, 15, 16}},
                                                                    {{17, 18, 19, 20},
                                                                     {21, 22, 23, 24}}};

int main(int argc, char *argv[])
{
    TensorPtr dataTensor(new HomogenTensor<float>(nDim, dims, (float *)dataArray));

    printTensor3d(dataTensor, "Forward maximum pooling layer input:");

    /* Create an algorithm to compute forward pooling layer results using maximum method */
    maximum_pooling3d::forward::Batch<> forwardLayer(nDim);
    forwardLayer.input.set(forward::data, dataTensor);

    /* Compute forward pooling layer results */
    forwardLayer.compute();

    /* Get the computed forward pooling layer results */
    services::SharedPtr<maximum_pooling3d::forward::Result> forwardResult = forwardLayer.getResult();

    printTensor3d(forwardResult->get(forward::value),
        "Forward maximum pooling layer result:");
    printTensor3d(forwardResult->get(maximum_pooling3d::auxSelectedIndices),
        "Forward maximum pooling layer selected indices:");

    /* Create an algorithm to compute backward pooling layer results using maximum method */
    maximum_pooling3d::backward::Batch<> backwardLayer(nDim);
    backwardLayer.input.set(backward::inputGradient, forwardResult->get(forward::value));
    backwardLayer.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward pooling layer results */
    backwardLayer.compute();

    /* Get the computed backward pooling layer results */
    services::SharedPtr<backward::Result> backwardResult = backwardLayer.getResult();

    printTensor3d(backwardResult->get(backward::gradient),
        "Backward maximum pooling layer result:");

    return 0;
}
