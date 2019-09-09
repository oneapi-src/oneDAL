/* file: loss_logistic_entr_layer_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
!    C++ example of forward and backward logistic cross-entropy layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LOSS_LOGISTIC_CROSS_ENTROPY_LAYER_BATCH"></a>
 * \example loss_logistic_entr_layer_dense_batch.cpp
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
string datasetName = "../data/batch/logistic_cross_entropy_layer.csv";
string datasetGroundTruthName = "../data/batch/logistic_cross_entropy_layer_ground_truth.csv";

int main()
{
    /* Read datasetFileName from a file and create a tensor to store input data */
    TensorPtr tensorData = readTensorFromCSV(datasetName);
    TensorPtr groundTruth = readTensorFromCSV(datasetGroundTruthName);

    /* Create an algorithm to compute forward logistic cross-entropy layer results using default method */
    loss::logistic_cross::forward::Batch<> logisticCrossEntropyLayerForward;

    /* Set input objects for the forward logistic cross-entropy layer */
    logisticCrossEntropyLayerForward.input.set(forward::data, tensorData);
    logisticCrossEntropyLayerForward.input.set(loss::forward::groundTruth, groundTruth);

    /* Compute forward logistic cross-entropy layer results */
    logisticCrossEntropyLayerForward.compute();

    /* Print the results of the forward logistic cross-entropy layer */
    loss::logistic_cross::forward::ResultPtr forwardResult = logisticCrossEntropyLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward logistic cross-entropy layer result (first 5 rows):", 5);
    printTensor(forwardResult->get(loss::logistic_cross::auxGroundTruth), "Logistic Cross-Entropy layer ground truth (first 5 rows):", 5);

    /* Create an algorithm to compute backward logistic cross-entropy layer results using default method */
    loss::logistic_cross::backward::Batch<> logisticCrossEntropyLayerBackward;

    /* Set input objects for the backward logistic cross-entropy layer */
    logisticCrossEntropyLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward logistic cross-entropy layer results */
    logisticCrossEntropyLayerBackward.compute();

    /* Print the results of the backward logistic cross-entropy layer */
    backward::ResultPtr backwardResult = logisticCrossEntropyLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward logistic cross-entropy layer result (first 5 rows):", 5);

    return 0;
}
