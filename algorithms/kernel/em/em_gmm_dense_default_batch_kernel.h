/* file: em_gmm_dense_default_batch_kernel.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//++
//  Declaration of template function that calculate ems.
//--
*/

#ifndef __EM_GMM_DENSE_DEFAULT_BATCH_KERNEL_H__
#define __EM_GMM_DENSE_DEFAULT_BATCH_KERNEL_H__

#include "em_gmm.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_blas.h"
#include "em_gmm_dense_default_batch_task.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
class EMKernel : public Kernel
{
public:
    typedef SharedPtr<GmmModel<algorithmFPType, cpu> > GmmModelPtr;
    EMKernel() {};

    services::Status compute(NumericTable &dataTable,
                             NumericTable &initialWeights,
                             NumericTable &initialMeans,
                             NumericTable **initialCovariances,
                             NumericTable &resultWeights,
                             NumericTable &resultMeans,
                             NumericTable **resultCovariances,
                             NumericTable &resultNIterations,
                             NumericTable &resultGoalFunction,
                             const Parameter &par);
};

template<typename algorithmFPType, CpuType cpu>
void stepM_mergePartialSums(
    algorithmFPType *cp_n, algorithmFPType *cp_m,
    algorithmFPType *mean_n, algorithmFPType *mean_m,
    algorithmFPType &w_n, algorithmFPType &w_m,
    size_t nFeatures, GmmModel<algorithmFPType, cpu> *covs);

template<typename algorithmFPType, Method method, CpuType cpu>
class EMKernelTask
{
    typedef SharedPtr<GmmModel<algorithmFPType, cpu> > GmmModelPtr;
    typedef GmmModelDiag<algorithmFPType, cpu> GmmModelDiagType;
    typedef GmmModelFull<algorithmFPType, cpu> GmmModelFullType;

    SharedPtr<GmmModel<algorithmFPType, cpu> > initializeCovariances();
public:
    EMKernelTask(NumericTable &dataTable,
                 NumericTable &initialWeights, NumericTable &initialMeans, NumericTable **initialCovariances,
                 NumericTable &resultWeights, NumericTable &resultMeans, NumericTable **resultCovariances,
                 NumericTable &resultNIterations,
                 NumericTable &resultGoalFunction,
                 const Parameter &par);

    services::Status compute();

    Status initialize();
    services::Status setStartValues();
    void setResultToZero();
    Status stepM_merge(size_t iteration);

    static void stepE(const size_t nVectorsInCurrentBlock, Task<algorithmFPType, cpu> &t, em_gmm::CovarianceStorageId covType);
    static algorithmFPType computePartialLogLikelyhood(const size_t nVectorsInCurrentBlock, Task<algorithmFPType, cpu> &t);
    static Status stepM_partial(const size_t nVectorsInCurrentBlock, Task<algorithmFPType, cpu> &t, em_gmm::CovarianceStorageId covType);
    static void stepM_mergePartialSums(
        algorithmFPType *cp_n, algorithmFPType *cp_m,
        algorithmFPType *mean_n, algorithmFPType *mean_m,
        algorithmFPType &w_n, algorithmFPType &w_m,
        size_t nFeatures, GmmModel<algorithmFPType, cpu> *covs);


    algorithmFPType *alpha;
    algorithmFPType *means;
    algorithmFPType *logAlpha;
    int *iterCounterArray;
    algorithmFPType *logLikelyhoodArray;

    size_t blockSizeDefault;
    size_t nBlocks;

    const DAAL_INT nFeatures;
    const DAAL_INT nVectors;
    const DAAL_INT nComponents;
    algorithmFPType logLikelyhoodCorrection;
    const DAAL_INT maxIterations;
    const algorithmFPType threshold;
    TArray<WriteRows<algorithmFPType, cpu, NumericTable>, cpu> covsPtr;
    GmmModelPtr covs;

    WriteRows<algorithmFPType, cpu, NumericTable> weightsBD;
    WriteRows<algorithmFPType, cpu, NumericTable> meansBD;
    WriteRows<int, cpu, NumericTable> nIterationsBD;
    WriteRows<algorithmFPType, cpu, NumericTable> goalFunctionBD;

    NumericTable &dataTable;
    NumericTable &initialWeights;
    NumericTable &initialMeans;
    NumericTable **initialCovariances;
    NumericTable &resultWeights;
    NumericTable &resultMeans;
    NumericTable **resultCovariances;
    NumericTable &resultNIterations;
    NumericTable &resultGoalFunction;
    const Parameter &par;
};

} // namespace internal

} // namespace em_gmm

} // namespace algorithms

} // namespace daal

#endif
