/* file: covariance_csr_distributed_mpi.cpp */
/*******************************************************************************
* Copyright 2017 Intel Corporation
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
!    C++ sample of sparse variance-covariance matrix computation in the
!    distributed processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-SAMPLE-CPP-COVARIANCE_CSR_DISTRIBUTED"></a>
 * \example covariance_csr_distributed_mpi.cpp
 */

#include <mpi.h>
#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Input data set parameters */
const size_t nBlocks = 4;

int rankId, comm_size;
#define mpi_root 0

const std::string datasetFileNames[] = { "./data/distributed/covcormoments_csr_1.csv",
                                         "./data/distributed/covcormoments_csr_2.csv",
                                         "./data/distributed/covcormoments_csr_3.csv",
                                         "./data/distributed/covcormoments_csr_4.csv" };

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &datasetFileNames[0],
                   &datasetFileNames[1],
                   &datasetFileNames[2],
                   &datasetFileNames[3]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    CSRNumericTable* dataTable = createSparseTable<float>(datasetFileNames[rankId]);

    /* Create an algorithm to compute a sparse variance-covariance matrix on local nodes */
    covariance::Distributed<step1Local, algorithmFPType, covariance::fastCSR> localAlgorithm;

    /* Set the input data set to the algorithm */
    localAlgorithm.input.set(covariance::data, CSRNumericTablePtr(dataTable));

    /* Compute a sparse variance-covariance matrix */
    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    services::SharedPtr<byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();

    /* Serialized data is of equal size on each node if each node called compute() equal number of times */
    if (rankId == mpi_root) {
        serializedData = services::SharedPtr<byte>(new byte[perNodeArchLength * nBlocks]);
    }

    byte* nodeResults = new byte[perNodeArchLength];
    dataArch.copyArchiveToArray(nodeResults, perNodeArchLength);

    /* Transfer partial results to step 2 on the root node */
    MPI_Gather(nodeResults,
               perNodeArchLength,
               MPI_CHAR,
               serializedData.get(),
               perNodeArchLength,
               MPI_CHAR,
               mpi_root,
               MPI_COMM_WORLD);

    delete[] nodeResults;

    if (rankId == mpi_root) {
        /* Create an algorithm to compute a sparse variance-covariance matrix on the master node */
        covariance::Distributed<step2Master, algorithmFPType, covariance::fastCSR> masterAlgorithm;

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() + perNodeArchLength * i,
                                       perNodeArchLength);

            covariance::PartialResultPtr dataForStep2FromStep1 =
                covariance::PartialResultPtr(new covariance::PartialResult());

            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm */
            masterAlgorithm.input.add(covariance::partialResults, dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute a sparse variance-covariance matrix on the master node */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        covariance::ResultPtr result = masterAlgorithm.getResult();

        /* Print the results */
        printNumericTable(result->get(covariance::covariance),
                          "Covariance matrix (upper left square 10*10) :",
                          10,
                          10);
        printNumericTable(result->get(covariance::mean), "Mean vector:", 1, 10);
    }

    MPI_Finalize();

    return 0;
}
