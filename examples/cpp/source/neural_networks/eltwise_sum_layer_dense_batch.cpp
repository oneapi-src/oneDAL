/* file: eltwise_sum_layer_dense_batch.cpp */
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
!    C++ example of forward and backward element-wise sum layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-ELTWISE_SUM_LAYER_BATCH"></a>
 * \example eltwise_sum_layer_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::data_management;
using namespace daal::services;

/* Input data set parameters */
string datasetName = "../data/batch/layer.csv";

/* Number of input tensors */
const size_t nInputs = 3;

int main()
{
    /* Create an algorithm to compute forward element-wise sum layer results using default method */
    eltwise_sum::forward::Batch<> eltwiseSumLayerForward;

    /* Read datasetFileName from a file and create a tensor to store input data */
    for (size_t i = 0; i < nInputs; i++)
    {
        TensorPtr tensorData = readTensorFromCSV(datasetName);

        /* Set input objects for the forward element-wise sum layer */
        eltwiseSumLayerForward.input.set(forward::inputLayerData, tensorData, i);
    }

    /* Compute forward element-wise sum layer results */
    eltwiseSumLayerForward.compute();

    /* Print the results of the forward element-wise sum layer */
    eltwise_sum::forward::ResultPtr forwardResult = eltwiseSumLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward element-wise sum layer result (first 5 rows):", 5);
    printNumericTable(forwardResult->get(eltwise_sum::auxNumberOfCoefficients),
        "Forward element-wise sum layer number of inputs (number of coefficients)", 1);

    /* Create an algorithm to compute backward element-wise sum layer results using default method */
    eltwise_sum::backward::Batch<> eltwiseSumLayerBackward;

    /* Set input objects for the backward element-wise sum layer */
    eltwiseSumLayerBackward.input.set(backward::inputGradient, readTensorFromCSV(datasetName));
    eltwiseSumLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward element-wise sum layer results */
    eltwiseSumLayerBackward.compute();

    /* Print the results of the backward element-wise sum layer */
    eltwise_sum::backward::ResultPtr backwardResult = eltwiseSumLayerBackward.getResult();

    for (size_t i = 0; i < nInputs; i++)
    {
        printTensor(backwardResult->get(backward::resultLayerData, i),
            "Backward element-wise sum layer backward result (first 5 rows):", 5);
    }

    return 0;
}
