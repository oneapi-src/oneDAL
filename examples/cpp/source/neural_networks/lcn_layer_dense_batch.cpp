/* file: lcn_layer_dense_batch.cpp */
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
!    C++ example of forward and backward local contrast normalization layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LCN_LAYER_BATCH"></a>
 * \example lcn_layer_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::data_management;
using namespace daal::services;

int main()
{
    /* Create collection of dimension sizes of the input data tensor */
    Collection<size_t> inDims;
    inDims.push_back(2);
    inDims.push_back(1);
    inDims.push_back(3);
    inDims.push_back(4);
    TensorPtr tensorData = TensorPtr(new HomogenTensor<>(inDims, Tensor::doAllocate, 1.0f));

    /* Create an algorithm to compute forward local contrast normalization layer results using default method */
    lcn::forward::Batch<> lcnLayerForward;

    /* Set input objects for the forward local contrast normalization layer */
    lcnLayerForward.input.set(forward::data, tensorData);

    /* Compute forward local contrast normalization layer results */
    lcnLayerForward.compute();

    /* Print the results of the forward local contrast normalization layer */
    lcn::forward::ResultPtr forwardResult = lcnLayerForward.getResult();
    printTensor(forwardResult->get(forward::value),       "Forward local contrast normalization layer result:");
    printTensor(forwardResult->get(lcn::auxCenteredData), "Centered data tensor:");
    printTensor(forwardResult->get(lcn::auxSigma),        "Sigma tensor:");
    printTensor(forwardResult->get(lcn::auxC),            "C tensor:");
    printTensor(forwardResult->get(lcn::auxInvMax),       "Inverted max(sigma, C):");

    /* Create input gradient tensor for backward local contrast normalization layer */
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<>(inDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward local contrast normalization layer results using default method */
    lcn::backward::Batch<> lcnLayerBackward;
    lcnLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    lcnLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward local contrast normalization layer results */
    lcnLayerBackward.compute();

    /* Get the computed backward local contrast normalization layer results */
    backward::ResultPtr backwardResult = lcnLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Local contrast normalization layer backpropagation gradient result:");

    return 0;
}
