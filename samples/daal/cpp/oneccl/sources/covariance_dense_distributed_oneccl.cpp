/* file: covariance_dense_distributed_mpi.cpp */
/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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
!    C++ sample of dense variance-covariance matrix computation in the
!    distributed processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-SAMPLE-CPP-COVARIANCE_DENSE_DISTRIBUTED"></a>
 * \example covariance_dense_distributed_mpi.cpp
 */

#include "daal_sycl.h"
#include "service.h"
#include "oneapi/ccl.hpp"
#include "mpi.h"
#include "stdio.h"
#include <memory>

#include <iostream>
#include "timing.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

const string datasetFileNames[] = { "./data/higgs_1m_train.csv",
                                    "./data/h1.csv", "./data/h2.csv",
                                    "./data/higgs_250t_test_1.csv", "./data/higgs_250t_test_2.csv",
                                    "./data/higgs_250t_test_3.csv", "./data/higgs_250t_test_4.csv"};

#define _P(...) do{ \
    printf(__VA_ARGS__); printf("\n"); fflush(0); \
    }while(0)

int rankId, comm_size;
const size_t nTimes = 5;
#define ccl_root  0

typedef services::SharedPtr<SyclHomogenNumericTable<>> SyclHomogenNumericTablePtr;

std::vector<sycl::device> get_gpus() {
    auto platforms = sycl::platform::get_platforms();
    for (auto p : platforms) {
        auto devices = p.get_devices(sycl::info::device_type::gpu);
        if (!devices.empty()) {
            return devices;
        }
    }
    return {};
}

int getLocalRank(ccl::communicator& comm, int size, int rank)
{
    char zero = static_cast<char>(0);
    std::vector<char> name(MPI_MAX_PROCESSOR_NAME + 1, zero);
    int resultlen = 0;
    MPI_Get_processor_name(name.data(), &resultlen);
    std::string str(name.begin(), name.end());
    std::vector<char> allNames((MPI_MAX_PROCESSOR_NAME + 1) * size, zero);
    std::vector<size_t> aReceiveCount(size, MPI_MAX_PROCESSOR_NAME + 1);
    ccl::allgatherv((int8_t*)name.data(), name.size(),  (int8_t*)allNames.data(), aReceiveCount, comm).wait();
    int localRank = 0;
    for(int i = 0; i < rank; i++) {
        auto nameBegin = allNames.begin() + i * (MPI_MAX_PROCESSOR_NAME + 1);
        std::string nbrName(nameBegin, nameBegin + (MPI_MAX_PROCESSOR_NAME + 1));
        if(nbrName == str)
            localRank++;
    }
    return localRank;
}

SyclHomogenNumericTablePtr loadData(int rankId, int size)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileNames[(size - 1) + rankId], DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    auto data = SyclHomogenNumericTable<>::create(10, 0, NumericTable::notAllocate);
    dataSource.loadDataBlock(data.get());
    return data;
}

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 7, &datasetFileNames[0], &datasetFileNames[1], &datasetFileNames[2], &datasetFileNames[3], &datasetFileNames[4], &datasetFileNames[5], &datasetFileNames[6]);

    ccl::init();

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    const size_t nBlocks = comm_size;

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rankId == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    auto comm = ccl::create_communicator(comm_size, rankId, kvs);
    auto local_rank = getLocalRank(comm, comm_size, rankId);

    auto gpus = get_gpus();
    auto rank_gpu = gpus[local_rank % gpus.size()];
    cl::sycl::queue queue(rank_gpu);

    daal::services::SyclExecutionContext ctx(queue);
    services::Environment::getInstance()->setDefaultExecutionContext(ctx);

    std::vector<double> times_local_comp(nTimes);
    std::vector<double> times_local_aux(nTimes);
    std::vector<double> times_master_comp(nTimes);
    std::vector<double> times_master_aux(nTimes);

    auto pData     = loadData(rankId, comm_size);

    for(size_t iter = 0; iter < nTimes; iter++)
    {
        timer local_comp_s = timer::start();

        /* Create an algorithm to compute a variance-covariance matrix on local nodes */
        covariance::Distributed<step1Local> localAlgorithm;

        /* Set the input data set to the algorithm */
        localAlgorithm.input.set(covariance::data, pData);

        /* Compute a variance-covariance matrix */
        localAlgorithm.compute();

        times_local_comp[iter] = timer::stop(local_comp_s);

        timer local_aux_s = timer::start();
        /* Serialize partial results required by step 2 */
        InputDataArchive dataArch;
        localAlgorithm.getPartialResult()->serialize(dataArch);
        size_t perNodeArchLength = dataArch.getSizeOfArchive();

        std::vector<size_t> aPerNodeArchLength(comm.size(), perNodeArchLength);

        const bool isRoot          = (rankId == ccl_root);

        /* Serialized data is of equal size on each node if each node called compute() equal number of times */
        ByteBuffer serializedData;
        serializedData.resize(perNodeArchLength * nBlocks);

        ByteBuffer nodeResults(perNodeArchLength);
        dataArch.copyArchiveToArray(&nodeResults[0], perNodeArchLength);

        /* Transfer partial results to step 2 on the root node */
        ccl::allgatherv((int8_t*)&nodeResults[0], perNodeArchLength,  (int8_t*)&serializedData[0], aPerNodeArchLength, comm).wait();

        times_local_aux[iter] = timer::stop(local_aux_s);

        if (isRoot)
        {
            timer master_aux_s = timer::start();

            /* Create an algorithm to compute a variance-covariance matrix on the master node */
            covariance::Distributed<step2Master> masterAlgorithm;
            masterAlgorithm.parameter.outputMatrixType = covariance::correlationMatrix;

            for (size_t i = 0; i < nBlocks; i++)
            {
                /* Deserialize partial results from step 1 */
                OutputDataArchive dataArch(&serializedData[perNodeArchLength * i], perNodeArchLength);

                covariance::PartialResultPtr dataForStep2FromStep1 = covariance::PartialResultPtr(new covariance::PartialResult());

                dataForStep2FromStep1->deserialize(dataArch);

                /* Set local partial results as input for the master-node algorithm */
                masterAlgorithm.input.add(covariance::partialResults, dataForStep2FromStep1);
            }
            times_master_aux[iter] = timer::stop(master_aux_s);

            timer master_comp_s = timer::start();

            /* Merge and finalizeCompute a dense variance-covariance matrix on the master node */
            masterAlgorithm.compute();
            masterAlgorithm.finalizeCompute();
            times_master_comp[iter] = timer::stop(master_comp_s);

            /* Retrieve the algorithm results */
            covariance::ResultPtr result = masterAlgorithm.getResult();

            if(iter == nTimes - 1)
            {
                /* Print the results */
                printNumericTable(result->get(covariance::covariance), "Covariance matrix:");
                printNumericTable(result->get(covariance::mean), "Mean vector:");
            }
        }
    }

    double f_loc_cmp_t = FirstIteration(times_local_comp);
    double f_loc_aux_t = FirstIteration(times_local_aux);
    double f_mst_cmp_t = FirstIteration(times_master_comp);
    double f_mst_aux_t = FirstIteration(times_master_aux);

    double loc_cmp_t = BoxFilter(times_local_comp);
    double loc_aux_t = BoxFilter(times_local_aux);
    double mst_cmp_t = BoxFilter(times_master_comp);
    double mst_aux_t = BoxFilter(times_master_aux);

    double f_time = f_loc_cmp_t + f_loc_aux_t + f_mst_cmp_t + f_mst_aux_t;
    double time = loc_cmp_t + loc_aux_t + mst_cmp_t + mst_aux_t;

    _P("size=%d f_time;time %.6f;%.6f", comm_size, f_time, time);
    _P("size=%d f_loc_cmp;f_loc_aux; f_mst_cmp;f_mst_aux %.6f;%.6f; %.6f;%.6f",
        comm_size, f_loc_cmp_t, f_loc_aux_t, f_mst_cmp_t, f_mst_aux_t);
    _P("size=%d loc_cmp;loc_aux; mst_cmp;mst_aux %.6f;%.6f; %.6f;%.6f",
        comm_size, loc_cmp_t, loc_aux_t, mst_cmp_t, mst_aux_t);

    MPI_Finalize();

    return 0;
}
