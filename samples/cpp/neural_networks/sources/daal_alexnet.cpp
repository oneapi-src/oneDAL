/* file: daal_alexnet.cpp */
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
!    C++ example of neural network training and scoring with AlexNet topology
!******************************************************************************/

#include "daal_commons.h"
#include "daal_alexnet.h"

const std::string defaultDatasetsPath = "./data";
const std::string datasetFileNames[] =
{
    "train_227x227.blob",
    "test_227x227.blob"
};

int main(int argc, char *argv[])
{
    std::string userDatasetsPath = getUserDatasetPath(argc, argv);
    std::string datasetsPath = selectDatasetPathOrExit(
        defaultDatasetsPath, userDatasetsPath, datasetFileNames, 2);

    /* Form path to the training and testing datasets */
    std::string trainBlobPath = datasetsPath + "/" + datasetFileNames[0];
    std::string testBlobPath  = datasetsPath + "/" + datasetFileNames[1];

    /* Create blob dataset reader for the training dataset (ImageBlobDatasetReader defined in blob_dataset.h)  */
    ImageBlobDatasetReader<float> trainDatasetReader(trainBlobPath, batchSize);
    training::TopologyPtr topology = configureNet(); /* defined in daal_alexnet.h */

    /* Train model (trainClassifier is defined in daal_common.h) */
    prediction::ModelPtr predictionModel = trainClassifier(topology, &trainDatasetReader);

    /* Create blob dataset reader for the testing dataset */
    ImageBlobDatasetReader<float> testDatasetReader(testBlobPath, batchSize);

    /* Test model (testClassifier is defined in daal_common.h) */
    float top5ErrorRate = testClassifier(predictionModel, &testDatasetReader);

    std::cout << "Top-5 error = " << top5ErrorRate * 100.0 << "%" << std::endl;

    return 0;
}
