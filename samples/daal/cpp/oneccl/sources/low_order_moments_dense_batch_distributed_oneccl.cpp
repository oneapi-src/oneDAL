/* file: low_order_moments_dense_batch_distributed_oneccl.cpp */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
!    C++ sample of computing low order moments in the distributed processing mode 
!
!******************************************************************************/

/**
 * <a name="DAAL-SAMPLE-CPP-LOW_ORDER_MOMENTS_DENSE_DISTRIBUTED"></a>
 * \example low_order_moments_dense_batch_distributed_oneccl.cpp
 */

#include "daal_sycl.h"
#include "mpi.h"
#include "oneapi/ccl.hpp"
#include "service_sycl.h"
#include "stdio.h"
#include <memory>

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef float algorithmFPType; /* Algorithm floating-point type */
typedef services::SharedPtr<SyclHomogenNumericTable<>>
    SyclHomogenNumericTablePtr;

/* Low order moments algorithm parameters */
const size_t nProcs = 4;

/* Input data set parameters */
const string dataFileNames[4] = {
    "./data/covcormoments_dense_1.csv", "./data/covcormoments_dense_2.csv",
    "./data/covcormoments_dense_3.csv", "./data/covcormoments_dense_4.csv"};

#define ccl_root 0

SyclHomogenNumericTablePtr loadData(int rankId) {
  /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data
   * from a .csv file */
  FileDataSource<CSVFeatureManager> dataSource(
      dataFileNames[rankId], DataSource::notAllocateNumericTable,
      DataSource::doDictionaryFromContext);

  /* Retrieve the data from the input file */
  auto data =
      SyclHomogenNumericTable<>::create(10, 0, NumericTable::notAllocate);
  dataSource.loadDataBlock(data.get());
  return data;
}

int main(int argc, char *argv[]) {
  /* Initialize oneCCL */
  ccl::init();
  checkArguments(argc, argv, 4, &dataFileNames[0], &dataFileNames[1],
                 &dataFileNames[2], &dataFileNames[3]);

  MPI_Init(NULL, NULL);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const bool isRoot = (rank == ccl_root);

  ccl::shared_ptr_class<ccl::kvs> kvs;
  ccl::kvs::address_type main_addr;
  if (rank == 0) {
    kvs = ccl::create_main_kvs();
    main_addr = kvs->get_address();
    MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0,
              MPI_COMM_WORLD);
  } else {
    MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0,
              MPI_COMM_WORLD);
    kvs = ccl::create_kvs(main_addr);
  }

  auto comm = ccl::create_communicator(size, rank, kvs);
  /* Create GPU device from local rank and set execution context if possible */
  set_gpu_by_rank_if_possible(comm, size, rank);

  /* Retrieve the input data */
  auto pData = loadData(rank);

  /* Create an algorithm to compute low order moments on local nodes */
  low_order_moments::Distributed<step1Local> localAlgorithm;

  /* Set the input data set to the algorithm */
  localAlgorithm.input.set(low_order_moments::data, pData);

  /* Compute low order moments */
  localAlgorithm.compute();

  /* Serialize partial results required by step 2 */
  ByteBuffer serializedData;
  InputDataArchive dataArch;
  localAlgorithm.getPartialResult()->serialize(dataArch);
  size_t perNodeArchLength = static_cast<size_t>(dataArch.getSizeOfArchive());

  /* Serialized data is of equal size on each node if each node called compute()
   * equal number of times */
  serializedData.resize(perNodeArchLength * nProcs);

  ByteBuffer nodeResults(perNodeArchLength);
  dataArch.copyArchiveToArray(&nodeResults[0], perNodeArchLength);
  std::vector<size_t> aReceiveCount(comm.size(), perNodeArchLength);
  /* Transfer partial results to step 2 on the root node */
  ccl::allgatherv((int8_t *)&nodeResults[0], perNodeArchLength,
                  (int8_t *)&serializedData[0], aReceiveCount, comm)
      .wait();

  if (isRoot) {
    /* Create an algorithm to compute low order moments on the master node */
    low_order_moments::Distributed<step2Master> masterAlgorithm;

    for (int i = 0; i < nProcs; i++) {
      /* Deserialize partial results from step 1 */
      OutputDataArchive dataArch(&serializedData[perNodeArchLength * i],
                                 perNodeArchLength);

      low_order_moments::PartialResultPtr dataForStep2FromStep1(
          new low_order_moments::PartialResult());

      dataForStep2FromStep1->deserialize(dataArch);

      /* Set local partial results as input for the master-node algorithm */
      masterAlgorithm.input.add(low_order_moments::partialResults,
                                dataForStep2FromStep1);
    }

    /* Merge and finalizeCompute low order moments on the master node */
    masterAlgorithm.compute();
    masterAlgorithm.finalizeCompute();

    /* Retrieve the algorithm results */
    low_order_moments::ResultPtr res = masterAlgorithm.getResult();

    /* Print the results */
    printNumericTable(res->get(low_order_moments::minimum), "Minimum:");
    printNumericTable(res->get(low_order_moments::maximum), "Maximum:");
    printNumericTable(res->get(low_order_moments::sum), "Sum:");
    printNumericTable(res->get(low_order_moments::sumSquares),
                      "Sum of squares:");
    printNumericTable(res->get(low_order_moments::sumSquaresCentered),
                      "Sum of squared difference from the means:");
    printNumericTable(res->get(low_order_moments::mean), "Mean:");
    printNumericTable(res->get(low_order_moments::secondOrderRawMoment),
                      "Second order raw moment:");
    printNumericTable(res->get(low_order_moments::variance), "Variance:");
    printNumericTable(res->get(low_order_moments::standardDeviation),
                      "Standard deviation:");
    printNumericTable(res->get(low_order_moments::variation), "Variation:");
  }

  MPI_Finalize();

  return 0;
}
