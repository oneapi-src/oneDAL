/* file: locallycon2d_layer_dense_batch.cpp */
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
!    C++ example of forward and backward 2D locally connected layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LOCALLYCON2D_LAYER_BATCH"></a>
 * \example locallycon2d_layer_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::data_management;
using namespace daal::services;

int main(int argc, char *argv[])
{
    /* Create collection of dimension sizes of the input data tensor */
    Collection<size_t> inDims;
    inDims << 2 << 2 << 6 << 8;

    TensorPtr dataTensor = TensorPtr(new HomogenTensor<>(inDims, Tensor::doAllocate, 1.0f));

    /* Create an algorithm to compute forward 2D locally connected layer results using default method */
    locallyconnected2d::forward::Batch<> locallyconnected2dLayerForward;
    locallyconnected2dLayerForward.input.set(forward::data, dataTensor);

    /* Compute forward 2D locally connected layer results */
    locallyconnected2dLayerForward.compute();

    /* Get the computed forward 2D locally connected layer results */
    locallyconnected2d::forward::ResultPtr forwardResult = locallyconnected2dLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward 2D locally connected layer result (first 5 rows):", 5, 15);
    printTensor(forwardResult->get(locallyconnected2d::auxWeights), "2D locally connected layer weights (first 5 rows):", 5, 15);

    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    /* Create input gradient tensor for backward 2D locally connected layer */
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward 2D locally connected layer results using default method */
    locallyconnected2d::backward::Batch<> locallyconnected2dLayerBackward;
    locallyconnected2dLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    locallyconnected2dLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward 2D locally connected layer results */
    locallyconnected2dLayerBackward.compute();

    /* Get the computed backward 2D locally connected layer results */
    backward::ResultPtr backwardResult = locallyconnected2dLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient),
                "2D locally connected layer backpropagation gradient result (first 5 rows):", 5, 15);
    printTensor(backwardResult->get(backward::weightDerivatives),
                "2D locally connected layer backpropagation weightDerivative result (first 5 rows):", 5, 15);
    printTensor(backwardResult->get(backward::biasDerivatives),
                "2D locally connected layer backpropagation biasDerivative result (first 5 rows):", 5, 15);
    return 0;
}
