/* file: daal_custom_loss_layer.cpp */
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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
!    C++ example of forward and backward custom loss layer usage
!
!******************************************************************************/

#include "daal.h"
#include "service.h"
#include "daal_custom_loss_layer.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::data_management;
using namespace daal::services;

const std::string defaultDatasetsPath = "./data";
const std::string datasetFileNames[] =
{
    "custom_loss_layer.csv",
    "custom_loss_layer_ground_truth.csv"
};

int main(int argc, char *argv[])
{
    std::string userDatasetsPath = getUserDatasetPath(argc, argv);
    std::string datasetsPath = selectDatasetPathOrExit(
        defaultDatasetsPath, userDatasetsPath, datasetFileNames, 2);

    /* Form path to the training and testing datasets */
    std::string trainDatasetPath = datasetsPath + "/" + datasetFileNames[0];
    std::string testDatasetPath  = datasetsPath + "/" + datasetFileNames[1];

    /* Read datasetFileName from a file and create a tensor to store input data */
    TensorPtr tensorData = readTensorFromCSV(trainDatasetPath);
    TensorPtr groundTruth = readTensorFromCSV(testDatasetPath);

    /* Create an algorithm to compute forward custom loss layer results using default method */
    new_loss_layer::ForwardBatch<> customLossLayerForward;

    /* Set input objects for the forward custom loss layer */
    customLossLayerForward.input.set(forward::data, tensorData);
    customLossLayerForward.input.set(loss::forward::groundTruth, groundTruth);

    /* Compute forward custom loss layer results */
    customLossLayerForward.compute();

    /* Print the results of the forward custom loss layer */
    new_loss_layer::ForwardResultPtr forwardResult = customLossLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward custom loss layer result (first 5 rows):", 5);
    printTensor(forwardResult->get(new_loss_layer::auxGroundTruth), "custom loss layer ground truth (first 5 rows):", 5);

    /* Create an algorithm to compute backward custom loss layer results using default method */
    new_loss_layer::BackwardBatch<> customLossLayerBackward;

    /* Set input objects for the backward custom loss layer */
    customLossLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward custom loss layer results */
    customLossLayerBackward.compute();

    /* Print the results of the backward custom loss layer */
    new_loss_layer::BackwardResultPtr backwardResult = customLossLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward custom loss layer result (first 5 rows):", 5);

    return 0;
}
