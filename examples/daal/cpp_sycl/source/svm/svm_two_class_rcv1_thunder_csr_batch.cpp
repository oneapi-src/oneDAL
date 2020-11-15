/* file: svm_two_class_rcv1_thunder_csr_batch.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
!    C++ example of two-class support vector machine (SVM) classification using
!    the Thunder method with DPC++ interfaces
!
!******************************************************************************/

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"

#if defined(__linux__)
    #include <sys/time.h>
    #include <time.h>
#endif

std::uint64_t get_time()
{
#if defined(__linux__)
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000000000 + t.tv_nsec;
#else
    #error OS other than Linux are not supported
#endif
}

#include <map>
#include <vector>

#ifdef TBB_USE
    #include "tbb/enumerable_thread_specific.h"
#endif

namespace bench
{
struct task_tls
{
    static const std::uint64_t MAX_KERNELS = 256;
    std::map<const char *, std::uint64_t> kernels;
    std::uint64_t current_kernel = 0;
    std::uint64_t time_kernels[MAX_KERNELS];

    task_tls & local();
    ~task_tls();
    void clear();
};

#ifdef TBB_USE
using enumerate_type = tbb::enumerable_thread_specific<task_tls>;
#else
using enumerate_type = task_tls;
#endif

class Profiler
{
public:
    static Profiler * get_instance();

    std::map<const char *, std::uint64_t> combine();

    enumerate_type & get_task();

    void clear();

    static std::uint64_t get_time();

private:
    Profiler();
    enumerate_type task;
};

task_tls & task_tls::local()
{
    return *this;
}

task_tls::~task_tls()
{
    clear();
}

void task_tls::clear()
{
    current_kernel = 0;
    kernels.clear();
}

Profiler * Profiler::get_instance()
{
    static Profiler instance;
    return &instance;
}

Profiler::Profiler() {}

std::map<const char *, std::uint64_t> Profiler::combine()
{
#ifdef TBB_USE
    auto res = task.combine([](task_tls x, task_tls y) {
        task_tls res;
        res.kernels = x.kernels;
        auto ym     = y.kernels;

        for (auto y_i : ym)
        {
            auto it = res.kernels.find(y_i.first);

            if (it == res.kernels.end())
            {
                res.kernels.insert(y_i);
            }
            else
            {
                it->second = std::max(it->second, y_i.second);
            }
        }

        return x;
    });
    return res.kernels;
#else
    return task.kernels;
#endif
}

enumerate_type & Profiler::get_task()
{
    return task;
}

void Profiler::clear()
{
    task.clear();
    task = enumerate_type(task_tls());
}

std::uint64_t Profiler::get_time()
{
#if defined(__linux__)
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000000000 + t.tv_nsec;
#else
    #error OS other than Linux are not supported
#endif
}

} // namespace bench

namespace daal
{
namespace internal
{
class ProfilerTask
{
public:
    ProfilerTask(const char * task_name);
    ~ProfilerTask();

private:
    const char * _task_name;
};

class Profiler
{
public:
    static ProfilerTask startTask(const char * task_name);
    static void endTask(const char * task_name);
};

ProfilerTask Profiler::startTask(const char * task_name)
{
    const std::uint64_t ns_start                       = bench::Profiler::get_time();
    auto & task_local                                  = bench::Profiler::get_instance()->get_task().local();
    task_local.time_kernels[task_local.current_kernel] = ns_start;
    task_local.current_kernel++;
    return daal::internal::ProfilerTask(task_name);
}

void Profiler::endTask(const char * task_name)
{
    const std::uint64_t ns_end = bench::Profiler::get_time();
    auto & task_local          = bench::Profiler::get_instance()->get_task().local();
    task_local.current_kernel--;
    const std::uint64_t times = ns_end - task_local.time_kernels[task_local.current_kernel];

    auto it = task_local.kernels.find(task_name);
    if (it == task_local.kernels.end())
    {
        task_local.kernels.insert({ task_name, times });
    }
    else
    {
        it->second += times;
    }
}

ProfilerTask::ProfilerTask(const char * task_name) : _task_name(task_name) {}

ProfilerTask::~ProfilerTask()
{
    Profiler::endTask(_task_name);
}

} // namespace internal
} // namespace daal

struct ProcessedData
{
    ProcessedData(const daal::data_management::NumericTablePtr & aux_table, const daal::data_management::NumericTablePtr & y_predict)
        : aux_table_(aux_table), y_predict_(y_predict)
    {
        dim = aux_table_->getNumberOfColumns();

        num_rows = y_predict_->getNumberOfRows();
        num_cols = y_predict_->getNumberOfColumns();

        aux_table_->getBlockOfRows(0, num_rows, daal::data_management::ReadWriteMode::readOnly, bd_aux_table_);
        ptr_aux_table = bd_aux_table_.getBlockPtr();

        y_predict_->getBlockOfRows(0, num_rows, daal::data_management::ReadWriteMode::readOnly, bd_y_predict_);
        ptr_y_predict = bd_y_predict_.getBlockPtr();
    }

    ~ProcessedData()
    {
        if (y_predict_)
        {
            y_predict_->releaseBlockOfRows(bd_y_predict_);
        }
        if (aux_table_)
        {
            aux_table_->releaseBlockOfRows(bd_aux_table_);
        }
    }

public:
    size_t num_rows;
    size_t num_cols;
    size_t dim;
    const double * ptr_aux_table;
    const double * ptr_y_predict;

private:
    daal::data_management::BlockDescriptor<double> bd_aux_table_;
    daal::data_management::BlockDescriptor<double> bd_y_predict_;
    const daal::data_management::NumericTablePtr & aux_table_;
    const daal::data_management::NumericTablePtr & y_predict_;
};

double compute_metric(const daal::data_management::NumericTablePtr & aux_table, const daal::data_management::NumericTablePtr & y_predict)
{
    ProcessedData data(aux_table, y_predict);

    const size_t size = data.num_rows * data.num_cols;
    double sum        = 0.0;

    for (size_t elem_idx = 0; elem_idx < size; ++elem_idx)
    {
        sum += ((data.ptr_y_predict[elem_idx] > 0 ? 1 : -1) == data.ptr_aux_table[elem_idx]);
    }

    const double accuracy = sum / size;
    return accuracy;
}

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

using daal::data_management::internal::SyclHomogenNumericTable;
using daal::services::internal::SyclExecutionContext;

/* Input data set parameters */
// string trainDatasetFileName = "/nfs/inn/proj/mkl/mirror/NN/DAAL_datasets/svm/rcv1_data_csr.csv";
// string trainLabelsFileName  = "/nfs/inn/proj/mkl/mirror/NN/DAAL_datasets/svm/rcv1_label_csr.csv";

// string testDatasetFileName = "/nfs/inn/proj/mkl/mirror/NN/DAAL_datasets/svm/rcv1_data_csr.csv";
// string testLabelsFileName  = "/nfs/inn/proj/mkl/mirror/NN/DAAL_datasets/svm/rcv1_label_csr.csv";

string trainDatasetFileName = "../data/csr_svm_bench/rcv1_data_csr.csv";
string trainLabelsFileName  = "../data/csr_svm_bench/rcv1_label_csr.csv";

string testDatasetFileName = "../data/csr_svm_bench/rcv1_data_csr.csv";
string testLabelsFileName  = "../data/csr_svm_bench/rcv1_label_csr.csv";

/* Parameters for the SVM kernel function */
kernel_function::KernelIfacePtr kernel;

/* Model object for the SVM algorithm */
svm::training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;

void trainModel();
void testModel();
void printResults();

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    for (const auto & deviceSelector : getListOfDevices())
    {
        const auto & nameDevice = deviceSelector.first;
        const auto & device     = deviceSelector.second;

        // if (nameDevice == "HOST") break;

        cl::sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";

        SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

        trainModel();
        testModel();
        printResults();
    }

    return 0;
}

void trainModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data
   * from a .csv file */
    FileDataSource<CSVFeatureManager> trainLabelsDataSource(trainLabelsFileName, DataSource::doAllocateNumericTable,
                                                            DataSource::doDictionaryFromContext);

    /* Create numeric table for training data */
    auto trainData = createSyclSparseTable<float>(trainDatasetFileName);

    /* Retrieve the data from the input file */
    trainLabelsDataSource.loadDataBlock();

    auto rbfkernel = services::SharedPtr<kernel_function::rbf::Batch<float, kernel_function::rbf::fastCSR> >(
        new kernel_function::rbf::Batch<float, kernel_function::rbf::fastCSR>());

    double eta                 = std::min(trainData->getNumberOfColumns(), 500lu);
    rbfkernel->parameter.sigma = std::sqrt(eta * 0.5);
    kernel                     = rbfkernel;

    svm::training::Batch<float, svm::training::thunder> algorithm;
    algorithm.parameter.kernel            = kernel;
    algorithm.parameter.C                 = 1000.0;
    algorithm.parameter.accuracyThreshold = 0.001;
    algorithm.parameter.tau               = 1e-6;
    algorithm.parameter.maxIterations     = 400;

    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainLabelsDataSource.getNumericTable());

    printf("[Data] nRows: %lu; nCols: %lu; nnz: %lu\n", trainData->getNumberOfRows(), trainData->getNumberOfColumns(), trainData->getDataSize());

    bench::Profiler::get_instance()->clear();

    /* Build the SVM model */
    auto t1 = get_time();
    algorithm.compute();
    auto t2 = get_time();

    auto kernels_profiler = bench::Profiler::get_instance()->combine();
    for (auto & kernel_info : kernels_profiler)
    {
        auto kernel_name     = std::string("kernel:") + kernel_info.first;
        const double time_ms = double(kernel_info.second) / 1e6;
        printf("%s: %.3lf\n", kernel_name.c_str(), time_ms);
    }

    bench::Profiler::get_instance()->clear();

    printf("[Train compute] time, sec: %lf\n", double(t2 - t1) / 1e6);

    printf("\n");
    printf("\n");

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel()
{
    /* Create Numeric Tables for testing data */
    auto testData = createSyclSparseTable<float>(testDatasetFileName);

    /* Create an algorithm object to predict SVM values */
    svm::prediction::Batch<> algorithm;
    algorithm.parameter.kernel = kernel;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model, trainingResult->get(classifier::training::model));

    /* Predict SVM values */
    auto t1 = get_time();
    algorithm.compute();
    auto t2 = get_time();

    auto kernels_profiler = bench::Profiler::get_instance()->combine();
    for (auto & kernel_info : kernels_profiler)
    {
        auto kernel_name     = std::string("kernel:") + kernel_info.first;
        const double time_ms = double(kernel_info.second) / 1e6;
        printf("%s: %.3lf\n", kernel_name.c_str(), time_ms);
    }
    bench::Profiler::get_instance()->clear();
    printf("[Predict compute] time, sec: %lf\n", double(t2 - t1) / 1e6);

    printf("\n");
    printf("\n");

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from
   * a .csv file */
    FileDataSource<CSVFeatureManager> testLabelsDataSource(testLabelsFileName, DataSource::doAllocateNumericTable,
                                                           DataSource::doDictionaryFromContext);
    /* Retrieve the data from input file */
    testLabelsDataSource.loadDataBlock();

    auto testNT    = testLabelsDataSource.getNumericTable();
    auto predictNT = predictionResult->get(classifier::prediction::prediction);

    printf("[ACCURACY]: %.3lf\n", compute_metric(testNT, predictNT));

    // printNumericTables<int, float>(testNT, predictNT, "Ground truth\t", "Classification results",
    //                                "SVM classification results (first 20 observations):", 20);
}
