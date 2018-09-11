/* file: qr_dense_distr.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
*******************************************************************************/

/*
!  Content:
!    C++ example of computing QR decomposition in the distributed processing
!    mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-QR_DISTRIBUTED"></a>
 * \example qr_dense_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const size_t nBlocks      = 4;

const string datasetFileNames[] =
{
    "../data/distributed/qr_1.csv",
    "../data/distributed/qr_2.csv",
    "../data/distributed/qr_3.csv",
    "../data/distributed/qr_4.csv"
};

void computestep1Local(size_t block);
void computeOnMasterNode();
void finalizeComputestep1Local(size_t block);

data_management::DataCollectionPtr dataFromStep1ForStep2[nBlocks];
data_management::DataCollectionPtr dataFromStep1ForStep3[nBlocks];
data_management::DataCollectionPtr dataFromStep2ForStep3[nBlocks];
NumericTablePtr R;
NumericTablePtr Qi[nBlocks];

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 4, &datasetFileNames[0], &datasetFileNames[1], &datasetFileNames[2], &datasetFileNames[3]);

    for (size_t i = 0; i < nBlocks; i++)
    {
        computestep1Local(i);
    }

    computeOnMasterNode();

    for (size_t i = 0; i < nBlocks; i++)
    {
        finalizeComputestep1Local(i);
    }

    /* Print the results */
    printNumericTable(Qi[0], "Part of orthogonal matrix Q from 1st node:", 10);
    printNumericTable(R    , "Triangular matrix R:");

    return 0;
}

void computestep1Local(size_t block)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileNames[block], DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the input data */
    dataSource.loadDataBlock();

    /* Create an algorithm to compute QR decomposition on the local node */
    qr::Distributed<step1Local> algorithm;

    algorithm.input.set( qr::data, dataSource.getNumericTable() );

    /* Compute QR decomposition */
    algorithm.compute();

    dataFromStep1ForStep2[block] = algorithm.getPartialResult()->get( qr::outputOfStep1ForStep2 );
    dataFromStep1ForStep3[block] = algorithm.getPartialResult()->get( qr::outputOfStep1ForStep3 );
}

void computeOnMasterNode()
{
    /* Create an algorithm to compute QR decomposition on the master node */
    qr::Distributed<step2Master> algorithm;

    for (size_t i = 0; i < nBlocks; i++)
    {
        algorithm.input.add( qr::inputOfStep2FromStep1, i, dataFromStep1ForStep2[i] );
    }

    /* Compute QR decomposition */
    algorithm.compute();

    qr::DistributedPartialResultPtr pres = algorithm.getPartialResult();
    KeyValueDataCollectionPtr inputForStep3FromStep2 = pres->get( qr::outputOfStep2ForStep3 );

    for (size_t i = 0; i < nBlocks; i++)
    {
        dataFromStep2ForStep3[i] = services::staticPointerCast<data_management::DataCollection, SerializationIface>((*inputForStep3FromStep2)[i]);
    }

    qr::ResultPtr res = algorithm.getResult();

    R = res->get(qr::matrixR);
}

void finalizeComputestep1Local(size_t block)
{
    /* Create an algorithm to compute QR decomposition on the master node */
    qr::Distributed<step3Local> algorithm;

    algorithm.input.set( qr::inputOfStep3FromStep1, dataFromStep1ForStep3[block] );
    algorithm.input.set( qr::inputOfStep3FromStep2, dataFromStep2ForStep3[block] );

    /* Compute QR decomposition */
    algorithm.compute();

    algorithm.finalizeCompute();

    qr::ResultPtr res = algorithm.getResult();

    Qi[block] = res->get(qr::matrixQ);
}
