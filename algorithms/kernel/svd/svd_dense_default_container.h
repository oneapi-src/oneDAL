/* file: svd_dense_default_container.h */
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
//  Implementation of svd calculation algorithm container.
//--
*/

//#include "svd.h"
#include "svd_types.h"
#include "svd_batch.h"
#include "svd_online.h"
#include "svd_distributed.h"
#include "svd_dense_default_kernel.h"
#include "kernel.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace svd
{

/**
 *  \brief Initialize list of cholesky kernels with implementations for supported architectures
 */
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SVDBatchKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    size_t na = input->size();
    size_t nr = result->size();

    NumericTable *a0 = static_cast<NumericTable *>(input->get(data).get());
    NumericTable **a = &a0;
    NumericTable *r[3];
    r[0] = static_cast<NumericTable *>(result->get(singularValues     ).get());
    r[1] = static_cast<NumericTable *>(result->get(leftSingularMatrix ).get());
    r[2] = static_cast<NumericTable *>(result->get(rightSingularMatrix).get());
    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::SVDBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, na, a, nr, r, par);
}

/**
 *  \brief Initialize list of cholesky kernels with implementations for supported architectures
 */
template<typename algorithmFPType, Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::OnlineContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SVDOnlineKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::~OnlineContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status OnlineContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    OnlinePartialResult *partialResult = static_cast<OnlinePartialResult *>(_pres);
    Parameter &svdPar = *(static_cast<Parameter *>(_par));

    size_t na = input->size();

    NumericTable *a0 = static_cast<NumericTable *>(input->get(data).get());
    NumericTable **a = &a0;

    size_t m = a0->getNumberOfColumns();
    size_t n = a0->getNumberOfRows();

    Status s = partialResult->addPartialResultStorage<algorithmFPType>(m, n, svdPar);
    if(!s)
        return s;

    size_t nr = 2;
    data_management::DataCollection *rCollection = static_cast<data_management::DataCollection *>(partialResult->get(outputOfStep1ForStep2).get());
    size_t np = rCollection->size();

    NumericTable *r[2] = {0, 0};

    if( svdPar.leftSingularMatrix != notRequired )
    {
        data_management::DataCollection *qCollection
            = static_cast<data_management::DataCollection *>(partialResult->get(outputOfStep1ForStep3).get());
        r[0] = static_cast<NumericTable *>((*qCollection)[np - 1].get());
    }
    r[1] = static_cast<NumericTable *>((*rCollection)[np - 1].get());

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::SVDOnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, na, a, nr, r, &svdPar);
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status OnlineContainer<algorithmFPType, method, cpu>::finalizeCompute()
{
    Parameter &svdPar = *(static_cast<Parameter *>(_par));
    OnlinePartialResult *partialResult = static_cast<OnlinePartialResult *>(_pres);
    Result *result = static_cast<Result *>(_res);

    data_management::DataCollection *qCollection = static_cast<data_management::DataCollection *>(partialResult->get(
                                                                                                      outputOfStep1ForStep3).get());
    data_management::DataCollection *rCollection = static_cast<data_management::DataCollection *>(partialResult->get(
                                                                                                      outputOfStep1ForStep2).get());
    size_t np = rCollection->size();

    daal::internal::TArray<NumericTable *, cpu> aPtr(np * 2);
    NumericTable **a = aPtr.get();

    for(size_t i = 0; i < np; i++)
    {
        a[i     ] = static_cast<NumericTable *>((*rCollection)[i].get());
        if( svdPar.leftSingularMatrix != notRequired )
        {
            a[i + np] = static_cast<NumericTable *>((*qCollection)[i].get());
        }
        else
        {
            a[i + np] = 0;
        }
    }
    size_t na = np * 2;

    size_t nr = 3;
    NumericTable *r[3];
    r[0] = static_cast<NumericTable *>(result->get(singularValues     ).get());
    r[1] = static_cast<NumericTable *>(result->get(leftSingularMatrix ).get());
    r[2] = static_cast<NumericTable *>(result->get(rightSingularMatrix).get());

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::SVDOnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), finalizeCompute, na, a, nr, r, &svdPar);
}

/**
 *  \brief Initialize list of cholesky kernels with implementations for supported architectures
 */
template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SVDDistributedStep2Kernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    DistributedStep2Input *input = static_cast<DistributedStep2Input *>(_in);
    DistributedPartialResult *partialResult = static_cast<DistributedPartialResult *>(_pres);

    data_management::KeyValueDataCollection *inCollection =
        static_cast<data_management::KeyValueDataCollection *>(input->get(inputOfStep2FromStep1).get());

    size_t nBlocks = input->getNBlocks();
    size_t nNodes  = inCollection->size();

    data_management::KeyValueDataCollection *perNodePartials = static_cast<data_management::KeyValueDataCollection *>(partialResult->get(outputOfStep2ForStep3).get());
    Result                 *results         = static_cast<Result *>(partialResult->get(finalResultFromStep2Master).get());

    size_t na = nBlocks;

    daal::internal::TArray<NumericTable *, cpu> aPtr(nBlocks);
    NumericTable **a = aPtr.get();

    size_t nr = nBlocks + 2;
    daal::internal::TArray<NumericTable *, cpu> rPtr(nBlocks + 2);
    NumericTable **r = rPtr.get();

    r[0] = static_cast<NumericTable *>(results->get(singularValues).get());
    r[1] = static_cast<NumericTable *>(results->get(rightSingularMatrix).get());

    size_t iBlocks = 0;
    for( size_t i = 0; i < nNodes; i++ )
    {
        data_management::DataCollection *nodeCollection = static_cast<data_management::DataCollection *>((*inCollection   ).getValueByIndex(i).get());
        data_management::DataCollection *nodePartials   = static_cast<data_management::DataCollection *>((*perNodePartials).getValueByIndex(i).get());

        size_t nodeSize = nodeCollection->size();

        for( size_t j = 0 ; j < nodeSize ; j++ )
        {
            a[iBlocks + j    ] = static_cast<NumericTable *>((*nodeCollection)[j].get());
            r[iBlocks + j + 2] = static_cast<NumericTable *>((*nodePartials  )[j].get());
        }

        iBlocks += nodeSize;
    }

    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::SVDDistributedStep2Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, na, a, nr, r, par);

    inCollection->clear();

    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step3Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SVDDistributedStep3Kernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step3Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status DistributedContainer<step3Local, algorithmFPType, method, cpu>::compute()
{
    DistributedStep3Input *input = static_cast<DistributedStep3Input *>(_in);
    DistributedPartialResultStep3 *partialResult = static_cast<DistributedPartialResultStep3 *>(_pres);

    data_management::DataCollectionPtr qCollection = input->get(inputOfStep3FromStep1);
    data_management::DataCollectionPtr rCollection = input->get(inputOfStep3FromStep2);

    ResultPtr result = partialResult->get(finalResultFromStep3);

    size_t nBlocks = qCollection->size();

    size_t na = nBlocks * 2;

    daal::internal::TArray<NumericTable *, cpu> aPtr(na);
    NumericTable **a = aPtr.get();

    for(size_t i = 0; i < nBlocks; i++)
    {
        a[i          ] = static_cast<NumericTable *>((*qCollection)[i].get());
        a[i + nBlocks] = static_cast<NumericTable *>((*rCollection)[i].get());
    }

    size_t nr = 1;
    NumericTable *r[1];
    r[0] = static_cast<NumericTable *>(result->get(leftSingularMatrix).get());

    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::SVDDistributedStep3Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, na, a, nr, r, par);
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status DistributedContainer<step3Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return Status();
}

}
}
} // namespace daal
