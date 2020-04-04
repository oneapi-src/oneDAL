
#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

// string trainDatasetFileName = "/nfs/inn/proj/numerics1/Users/kpetrov/ats/svm/svm_repo_impl/data.csv";
string trainDatasetFileName = "../data/batch/svm_two_class_train_dense.csv";

string testDatasetFileName = "../data/batch/svm_two_class_test_dense.csv";

const size_t nFeatures = 20;

// const size_t nFeatures = 119;

<<<<<<< HEAD
=======
svm::training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;
NumericTablePtr testGroundTruth;

void trainModel();
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
        if (!device.is_gpu()) continue;
        cl::sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";

        daal::services::SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

<<<<<<< HEAD
        trainModel(trainingResult);
=======
        trainModel();
        // testModel();
        // printResults();
>>>>>>> 8a074e5f... workin training
    }

    return 0;
}

void trainModel()
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

    svm::training::Batch<> algorithm;

    algorithm.parameter.kernel            = kernel;
    algorithm.parameter.cacheSize         = 40000000;
    algorithm.parameter.C                 = 1.0;
    algorithm.parameter.maxIterations     = 1000;
    algorithm.parameter.accuracyThreshold = 0.1;

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
    const size_t nSV             = svCoeffTable->getNumberOfRows();

    const float bias(model->getBias());

    printf("nSV %lu\n", nSV);
    printf("bias %lf\n", bias);
    printNumeric<float>(svCoeffTable, "", "svCoeffTable", 25);
    printNumeric<int>(svIndices, "", "svIndices", 25);
}

void testModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    testGroundTruth = NumericTablePtr(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
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
