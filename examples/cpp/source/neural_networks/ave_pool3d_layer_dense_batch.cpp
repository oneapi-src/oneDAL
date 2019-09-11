/* file: ave_pool3d_layer_dense_batch.cpp */
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
!    C++ example of neural network forward and backward three-dimensional average pooling layers usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-AVERAGE_POOLING3D_LAYER_BATCH"></a>
 * \example ave_pool3d_layer_dense_batch.cpp
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
    TensorPtr dataTensor(new HomogenTensor<>(nDim, dims, (float *)dataArray));

    printTensor3d(dataTensor, "Forward average pooling layer input:");

    /* Create an algorithm to compute forward pooling layer results using average method */
    average_pooling3d::forward::Batch<> forwardLayer(nDim);
    forwardLayer.input.set(forward::data, dataTensor);

    /* Compute forward pooling layer results */
    forwardLayer.compute();

    /* Get the computed forward pooling layer results */
    average_pooling3d::forward::ResultPtr forwardResult = forwardLayer.getResult();

    printTensor3d(forwardResult->get(forward::value),
        "Forward average pooling layer result:");
    printNumericTable(forwardResult->get(average_pooling3d::auxInputDimensions), "Forward pooling layer input dimensions:");

    /* Create an algorithm to compute backward pooling layer results using average method */
    average_pooling3d::backward::Batch<> backwardLayer(nDim);
    backwardLayer.input.set(backward::inputGradient, forwardResult->get(forward::value));
    backwardLayer.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward pooling layer results */
    backwardLayer.compute();

    /* Get the computed backward pooling layer results */
    backward::ResultPtr backwardResult = backwardLayer.getResult();

    printTensor3d(backwardResult->get(backward::gradient),
        "Backward average pooling layer result:");

    return 0;
}
