/* file: daal_googlenet_v1.cpp */
/*******************************************************************************
* Copyright 2017-2018 Intel Corporation.
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
*
* License:
* http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
* eement/
*******************************************************************************/

/*
!  Content:
!    C++ example of neural network training and scoring with GoogleNetV1 topology
!******************************************************************************/

#include "daal_commons.h"
#include "daal_googlenet_v1.h"

const std::string defaultDatasetsPath = "./data";
const std::string datasetFileNames[] =
{
    "train_224x224.blob",
    "test_224x224.blob"
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
    training::TopologyPtr topology = configureNet(); /* defined in daal_googlenet_v1.h */

    /* Train model (trainClassifier is defined in daal_common.h) */
    prediction::ModelPtr predictionModel = trainClassifier(topology, &trainDatasetReader);

    /* Create blob dataset reader for the testing dataset */
    ImageBlobDatasetReader<float> testDatasetReader(testBlobPath, batchSize);

    /* Test model (testClassifier is defined in daal_common.h) */
    float top5ErrorRate = testClassifier(predictionModel, &testDatasetReader);

    std::cout << "Top-5 error = " << top5ErrorRate * 100.0 << "%" << std::endl;

    return 0;
}
