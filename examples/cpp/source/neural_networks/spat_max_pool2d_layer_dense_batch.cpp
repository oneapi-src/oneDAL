/* file: spat_max_pool2d_layer_dense_batch.cpp */
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
!    C++ example of neural network forward and backward two-dimensional spatial pyramid maximum pooling layers usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SPAT_MAX_POOL2D_LAYER_DENSE_BATCH"></a>
 * \example spat_max_pool2d_layer_dense_batch.cpp
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
                                  {{{ -1, -2, -3, -4},
                                    { -5, -6, -7, -8}},
                                                    {{ -9, -10, -11, -12},
                                                    {-13, -14, -15, -16}},
                                                                    {{-17, -18, -19, -20},
                                                                     {-21, -22, -23, -24}}}};

int main(int argc, char *argv[])
{
    TensorPtr data(new HomogenTensor<>(nDim, dims, (float *)dataArray));

    printTensor(data, "Forward two-dimensional spatial pyramid maximum pooling layer input (first 10 rows):", 10);

    /* Create an algorithm to compute forward two-dimensional spatial pyramid maximum pooling layer results using default method */
    spatial_maximum_pooling2d::forward::Batch<> forwardLayer(2, nDim);
    forwardLayer.input.set(forward::data, data);

    /* Compute forward two-dimensional spatial pyramid maximum pooling layer results */
    forwardLayer.compute();

    /* Get the computed forward two-dimensional spatial pyramid maximum pooling layer results */
    spatial_maximum_pooling2d::forward::ResultPtr forwardResult = forwardLayer.getResult();

    printTensor(forwardResult->get(forward::value), "Forward two-dimensional spatial pyramid maximum pooling layer result (first 5 rows):", 5);
    printTensor(forwardResult->get(spatial_maximum_pooling2d::auxSelectedIndices),
        "Forward two-dimensional spatial pyramid maximum pooling layer selected indices (first 10 rows):", 10);


    /* Create an algorithm to compute backward two-dimensional spatial pyramid maximum pooling layer results using default method */
    spatial_maximum_pooling2d::backward::Batch<> backwardLayer(2, nDim);
    backwardLayer.input.set(backward::inputGradient, forwardResult->get(forward::value));
    backwardLayer.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));


    /* Compute backward two-dimensional spatial pyramid maximum pooling layer results */
    backwardLayer.compute();

    /* Get the computed backward two-dimensional spatial pyramid maximum pooling layer results */
    backward::ResultPtr backwardResult = backwardLayer.getResult();

    printTensor(backwardResult->get(backward::gradient),
        "Backward two-dimensional spatial pyramid maximum pooling layer result (first 10 rows):", 10);

    return 0;
}
