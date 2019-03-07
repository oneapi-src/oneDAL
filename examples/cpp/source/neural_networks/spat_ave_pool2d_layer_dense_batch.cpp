/* file: spat_ave_pool2d_layer_dense_batch.cpp */
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
    TensorPtr data(new HomogenTensor<>(nDim, dims, (float *)dataArray));
    printTensor(data, "Forward two-dimensional spatial pyramid average pooling layer input (first 10 rows):", 10);

    /* Create an algorithm to compute forward two-dimensional spatial pyramid average pooling layer results using default method */
    spatial_average_pooling2d::forward::Batch<> forwardLayer(2, nDim);
    forwardLayer.input.set(forward::data, data);

    /* Compute forward two-dimensional spatial pyramid average pooling layer results */
    forwardLayer.compute();

    /* Get the computed forward two-dimensional spatial pyramid average pooling layer results */
    spatial_average_pooling2d::forward::ResultPtr forwardResult = forwardLayer.getResult();

    printTensor(forwardResult->get(forward::value), "Forward two-dimensional spatial pyramid average pooling layer result (first 5 rows):", 5);
    printNumericTable(forwardResult->get(spatial_average_pooling2d::auxInputDimensions), "Forward two-dimensional spatial pyramid average pooling layer input dimensions:");

    /* Create an algorithm to compute backward two-dimensional spatial pyramid average pooling layer results using default method */
    spatial_average_pooling2d::backward::Batch<> backwardLayer(2, nDim);
    backwardLayer.input.set(backward::inputGradient, forwardResult->get(forward::value));
    backwardLayer.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward two-dimensional spatial pyramid average pooling layer results */
    backwardLayer.compute();

    /* Get the computed backward two-dimensional spatial pyramid average pooling layer results */
    backward::ResultPtr backwardResult = backwardLayer.getResult();

    printTensor(backwardResult->get(backward::gradient),
        "Backward two-dimensional spatial pyramid average pooling layer result (first 10 rows):", 10);

    return 0;
}
