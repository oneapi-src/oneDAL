/* file: kmeans_init_kernel.h */
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
//  Declaration of template function that computes K-means.
//--
*/

#ifndef _KMEANS_INIT_H
#define _KMEANS_INIT_H

#include "kmeans_init_types.h"
#include "kernel.h"
#include "numeric_table.h"
#include "memory_block.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace internal
{

#define isPlusPlusMethod(method)\
    ((method == kmeans::init::plusPlusDense) || (method == kmeans::init::plusPlusCSR) || \
    (method == kmeans::init::parallelPlusDense) || (method == kmeans::init::parallelPlusCSR))

template <Method method, typename algorithmFPType, CpuType cpu>
class KMeansInitKernel: public Kernel
{
public:
    services::Status compute(size_t na, const NumericTable *const *a, size_t nr, const NumericTable *const *r, const Parameter *par, engines::BatchBase &engine);
};

template <typename algorithmFPType, CpuType cpu>
class KMeansInitKernel<plusPlusDense, algorithmFPType, cpu> : public Kernel
{
public:
    services::Status compute(size_t na, const NumericTable *const *a, size_t nr, const NumericTable *const *r, const Parameter *par, engines::BatchBase &engine);
};

template <typename algorithmFPType, CpuType cpu>
class KMeansInitKernel<parallelPlusDense, algorithmFPType, cpu> : public Kernel
{
public:
    services::Status compute(size_t na, const NumericTable *const *a, size_t nr, const NumericTable *const *r, const Parameter *par, engines::BatchBase &engine);
};

template <typename algorithmFPType, CpuType cpu>
class KMeansInitKernel<plusPlusCSR, algorithmFPType, cpu> : public Kernel
{
public:
    services::Status compute(size_t na, const NumericTable *const *a, size_t nr, const NumericTable *const *r, const Parameter *par, engines::BatchBase &engine);
};

template <typename algorithmFPType, CpuType cpu>
class KMeansInitKernel<parallelPlusCSR, algorithmFPType, cpu> : public Kernel
{
public:
    services::Status compute(size_t na, const NumericTable *const *a, size_t nr, const NumericTable *const *r, const Parameter *par, engines::BatchBase &engine);
};

template <Method method, typename algorithmFPType, CpuType cpu>
class KMeansInitStep1LocalKernel: public Kernel
{
public:
    services::Status compute(const NumericTable* pData, const Parameter *par,
        NumericTable* pNumPartialClusters, NumericTablePtr& pPartialClusters, engines::BatchBase &engine);
};

template <typename algorithmFPType, CpuType cpu>
class KMeansInitPlusPlusStep1LocalKernel : public Kernel
{
public:
    services::Status compute(const NumericTable* pData, const Parameter *par,
        const NumericTable* pNumPartialClusters, NumericTable*& pPartialClusters, engines::BatchBase &engine);
};

template <Method method, typename algorithmFPType, CpuType cpu>
class KMeansInitStep2MasterKernel: public Kernel
{
public:
    services::Status finalizeCompute(size_t na, const NumericTable *const *a, NumericTable* ntClusters, const Parameter *par);
};

template <Method method, typename algorithmFPType, CpuType cpu>
class KMeansInitStep2LocalKernel : public Kernel
{
public:
    services::Status compute(const DistributedStep2LocalPlusPlusParameter* par,
        const NumericTable* pData, const NumericTable* pNewCenters, NumericTable** aLocalData,
        NumericTable* pRes, NumericTable* pOutputForStep5);
};

template <Method method, typename algorithmFPType, CpuType cpu>
class KMeansInitStep3MasterKernel : public Kernel
{
public:
    KMeansInitStep3MasterKernel() : _isFirstIteration(true) {}
    ~KMeansInitStep3MasterKernel() {}
    services::Status compute(const Parameter* par, const KeyValueDataCollection* pInputColl,
        MemoryBlock* pRngState,
        KeyValueDataCollection* pOutputColl, engines::BatchBase &engine);

protected:
    services::Status init(const Parameter* par, MemoryBlock* pRngState, engines::BatchBase &engine);

protected:
    bool _isFirstIteration;
    MemoryBlock* _rngState;
};

template <Method method, typename algorithmFPType, CpuType cpu>
class KMeansInitStep4LocalKernel : public Kernel
{
public:
    services::Status compute(const NumericTable* pData, const NumericTable* pInput, NumericTable** aLocalData, NumericTable* pOutput);
};

template <Method method, typename algorithmFPType, CpuType cpu>
class KMeansInitStep5MasterKernel : public Kernel
{
public:
    services::Status compute(const data_management::DataCollection* pCandidates,
        const data_management::DataCollection* pRating,
        NumericTable* pResCand, NumericTable* pResRating);
    services::Status finalizeCompute(const Parameter *par, const NumericTable* ntCand, const NumericTable* ntWeights,
        const MemoryBlock* pRngState, NumericTable* pCentroids, engines::BatchBase &engine);
};

} // namespace daal::algorithms::kmeans::init::internal
} // namespace daal::algorithms::kmeans::init
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal

#endif
