
#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

string trainDatasetFileName = "../data/batch/svm_two_class_train_dense.csv";
string testDatasetFileName  = "../data/batch/svm_two_class_test_dense.csv";

const size_t nFeatures = 20;

<<<<<<< HEAD
// const size_t nFeatures = 119;

<<<<<<< HEAD
=======
=======
>>>>>>> c8ef0452... stabile and working version
svm::training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;
NumericTablePtr testGroundTruth;

template <typename algorithmType>
void trainModel(algorithmType algorithm);
void testModel();
void printResults();

>>>>>>> 8a074e5f... workin training
kernel_function::KernelIfacePtr kernel(new kernel_function::linear::Batch<>());

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    for (const auto & deviceSelector : getListOfDevices())
    {
        const auto & nameDevice = deviceSelector.first;
        const auto & device     = deviceSelector.second;
        if (!(device.is_gpu() || device.is_cpu())) continue;
        cl::sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";

        daal::services::SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

<<<<<<< HEAD
<<<<<<< HEAD
        trainModel(trainingResult);
=======
        trainModel();
        // testModel();
        // printResults();
>>>>>>> 8a074e5f... workin training
=======
        if (device.is_gpu())
            trainModel(svm::training::Batch<float, svm::training::thunder>());
        else
            trainModel(svm::training::Batch<float, svm::training::boser>());
        testModel();
        printResults();
>>>>>>> c8ef0452... stabile and working version
    }

    return 0;
}

template <typename algorithmType>
void trainModel(algorithmType algorithm)
{
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    auto trainData        = SyclHomogenNumericTable<>::create(nFeatures, 0, NumericTable::doNotAllocate);
    auto trainGroundTruth = SyclHomogenNumericTable<>::create(1, 0, NumericTable::doNotAllocate);

<<<<<<< HEAD
<<<<<<< HEAD
    auto mergedData(new MergedNumericTable(trainData, trainGroundTruth));
=======
    NumericTablePtr mergedData(new MergedNumericTable(trainGroundTruth, trainData));
>>>>>>> 815734e6... fix build
=======
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainGroundTruth));
>>>>>>> ca30c048... add sort

    trainDataSource.loadDataBlock(mergedData.get());

    algorithm.parameter.kernel            = kernel;
    algorithm.parameter.C                 = 1.0;
    algorithm.parameter.accuracyThreshold = 0.01;
    algorithm.parameter.tau               = 1e-6;

    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);

    /* Build the SVM model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
<<<<<<< HEAD
=======

    auto model                   = trainingResult->get(classifier::training::model);
    NumericTablePtr svCoeffTable = model->getClassificationCoefficients();
    NumericTablePtr svIndices    = model->getSupportIndices();
    NumericTablePtr sv           = model->getSupportVectors();
    const size_t nSV             = svCoeffTable->getNumberOfRows();

    const float bias(model->getBias());

    printf("nSV %lu\n", nSV);
    printf("bias %lf\n", bias);
    // printNumeric<float>(svCoeffTable, "", "svCoeffTable", 25);
    // printNumeric<float>(svCoeffTable, "", "svCoeffTable", nSV);
    // printNumeric<int>(svIndices, "", "svIndices", 5);
    // printNumeric<float>(sv, "", "sv", 25);
}

void testModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData = SyclHomogenNumericTable<>::create(nFeatures, 0, NumericTable::doNotAllocate);
    testGroundTruth          = SyclHomogenNumericTable<>::create(1, 0, NumericTable::doNotAllocate);
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to predict SVM values */
    svm::prediction::Batch<> algorithm;

    algorithm.parameter.kernel = kernel;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model, trainingResult->get(classifier::training::model));

    /* Predict SVM values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults()
{
    printNumericTables<int, float>(testGroundTruth, predictionResult->get(classifier::prediction::prediction), "Ground truth\t",
                                   "Classification results", "SVM classification results (first 20 observations):", 20);
>>>>>>> 8a074e5f... workin training
}
