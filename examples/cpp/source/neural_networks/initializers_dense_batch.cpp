/* file: initializers_dense_batch.cpp */
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
!    C++ example of initializers
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-INITIALIZERS_DENSE_BATCH"></a>
 * \example initializers_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::data_management;
using namespace daal::services;

int main(int argc, char *argv[])
{
    /* Create collection of dimension sizes of the input data tensor */
    Collection<size_t> inDims;
    inDims.push_back(2);
    inDims.push_back(1);
    inDims.push_back(3);
    inDims.push_back(4);
    TensorPtr tensorData = TensorPtr(new HomogenTensor<>(inDims, Tensor::doAllocate));

    /* Fill tensor data using truncated gaussian initializer */
    /* Create an algorithm to initialize data using default method */
    initializers::truncated_gaussian::Batch<> truncatedGaussInitializer(0.0, 1.0);

    /* Set input object and parameters for the truncated gaussian initializer */
    truncatedGaussInitializer.input.set(initializers::data, tensorData);

    /* Compute truncated gaussian initializer */
    truncatedGaussInitializer.compute();

    /* Print the results of the truncated gaussian initializer */
    printTensor(tensorData, "Data with truncated gaussian distribution:");


    /* Fill tensor data using gaussian initializer */
    /* Create an algorithm to initialize data using default method */
    initializers::gaussian::Batch<> gaussInitializer(1.0, 0.5);

    /* Set input object for the gaussian initializer */
    gaussInitializer.input.set(initializers::data, tensorData);

    /* Compute gaussian initializer */
    gaussInitializer.compute();

    /* Print the results of the gaussian initializer */
    printTensor(tensorData, "Data with gaussian distribution:");


    /* Fill tensor data using uniform initializer */
    /* Create an algorithm to initialize data using default method */
    initializers::uniform::Batch<> uniformInitializer(-5.0, 5.0);

    /* Set input object and parameters for the uniform initializer */
    uniformInitializer.input.set(initializers::data, tensorData);

    /* Compute uniform initializer */
    uniformInitializer.compute();

    /* Print the results of the uniform initializer */
    printTensor(tensorData, "Data with uniform distribution:");


    /* Fill layer weights using xavier initializer */
    /* Create an algorithm to compute forward fully-connected layer results using default method */
    fullyconnected::forward::Batch<> fullyconnectedLayerForward(5);

    /* Set input objects and parameter for the forward fully-connected layer */
    fullyconnectedLayerForward.input.set(forward::data, tensorData);
    fullyconnectedLayerForward.parameter.weightsInitializer.reset(new initializers::xavier::Batch<>());

    /* Compute forward fully-connected layer results */
    fullyconnectedLayerForward.compute();

    /* Print the results of the xavier initializer */
    printTensor(fullyconnectedLayerForward.input.get(layers::forward::weights), "Weights filled by xavier initializer:");

    return 0;
}
