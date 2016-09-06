/* file: qr_dense_default_container.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//++
//  Implementation of qr calculation algorithm container.
//--
*/

//#include "qr.h"
#include "qr_types.h"
#include "qr_batch.h"
#include "qr_online.h"
#include "qr_distributed.h"
#include "qr_dense_default_kernel.h"

namespace daal
{
namespace algorithms
{
namespace qr
{

/**
 *  \brief Initialize list of cholesky kernels with implementations for supported architectures
 */
template<typename interm, Method method, CpuType cpu>
BatchContainer<interm, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::QRBatchKernel, interm, method);
}

template<typename interm, Method method, CpuType cpu>
BatchContainer<interm, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename interm, Method method, CpuType cpu>
void BatchContainer<interm, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    size_t na = input->size();
    size_t nr = result->size();

    NumericTable *a0 = static_cast<NumericTable *>(input->get(data).get());
    NumericTable **a = &a0;
    NumericTable *r[2];
    r[0] = static_cast<NumericTable *>(result->get(matrixQ).get());
    r[1] = static_cast<NumericTable *>(result->get(matrixR).get());
    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::QRBatchKernel, __DAAL_KERNEL_ARGUMENTS(interm, method), compute, na, a, nr, r, par);
}

/**
 *  \brief Initialize list of cholesky kernels with implementations for supported architectures
 */
template<typename interm, Method method, CpuType cpu>
OnlineContainer<interm, method, cpu>::OnlineContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::QROnlineKernel, interm, method);
}

template<typename interm, Method method, CpuType cpu>
OnlineContainer<interm, method, cpu>::~OnlineContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename interm, Method method, CpuType cpu>
void OnlineContainer<interm, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    OnlinePartialResult *partialResult = static_cast<OnlinePartialResult *>(_pres);

    size_t na = input->size();

    NumericTable *a0 = static_cast<NumericTable *>(input->get(data).get());
    NumericTable **a = &a0;

    size_t m = a0->getNumberOfColumns();
    size_t n = a0->getNumberOfRows();

    partialResult->addPartialResultStorage<interm>(m, n);

    size_t nr = 2;
    data_management::DataCollection *qCollection = static_cast<data_management::DataCollection *>(partialResult->get(
                                                                                                      outputOfStep1ForStep3).get());
    data_management::DataCollection *rCollection = static_cast<data_management::DataCollection *>(partialResult->get(
                                                                                                      outputOfStep1ForStep2).get());
    size_t np = qCollection->size();

    NumericTable *r[2];
    r[0] = static_cast<NumericTable *>((*qCollection)[np - 1].get());
    r[1] = static_cast<NumericTable *>((*rCollection)[np - 1].get());

    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::QROnlineKernel, __DAAL_KERNEL_ARGUMENTS(interm, method), compute, na, a, nr, r, par);
}

template<typename interm, Method method, CpuType cpu>
void OnlineContainer<interm, method, cpu>::finalizeCompute()
{
    OnlinePartialResult *partialResult = static_cast<OnlinePartialResult *>(_pres);
    Result *result = static_cast<Result *>(_res);

    data_management::DataCollection *qCollection = static_cast<data_management::DataCollection *>(partialResult->get(
                                                                                                      outputOfStep1ForStep3).get());
    data_management::DataCollection *rCollection = static_cast<data_management::DataCollection *>(partialResult->get(
                                                                                                      outputOfStep1ForStep2).get());
    size_t np = qCollection->size();

    NumericTable **a = new NumericTable*[np * 2];
    for(size_t i = 0; i < np; i++)
    {
        a[i     ] = static_cast<NumericTable *>((*rCollection)[i].get());
        a[i + np] = static_cast<NumericTable *>((*qCollection)[i].get());
    }
    size_t na = np * 2;

    size_t nr = result->size();
    NumericTable *r[2];
    r[0] = static_cast<NumericTable *>(result->get(matrixQ).get());
    r[1] = static_cast<NumericTable *>(result->get(matrixR).get());

    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::QROnlineKernel, __DAAL_KERNEL_ARGUMENTS(interm, method), finalizeCompute, na, a, nr, r, par);

    delete[] a;
}

/**
 *  \brief Initialize list of cholesky kernels with implementations for supported architectures
 */
template<typename interm, Method method, CpuType cpu>
DistributedContainer<step2Master, interm, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::QRDistributedStep2Kernel, interm, method);
}

template<typename interm, Method method, CpuType cpu>
DistributedContainer<step2Master, interm, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename interm, Method method, CpuType cpu>
void DistributedContainer<step2Master, interm, method, cpu>::compute()
{
    DistributedStep2Input *input = static_cast<DistributedStep2Input *>(_in);
    DistributedPartialResult *partialResult = static_cast<DistributedPartialResult *>(_pres);

    data_management::KeyValueDataCollection *inCollection =
        static_cast<data_management::KeyValueDataCollection *>(input->get(inputOfStep2FromStep1).get());

    size_t nBlocks = input->getNBlocks();
    size_t nNodes  = inCollection->size();

    data_management::KeyValueDataCollection *perNodePartials =
        static_cast<data_management::KeyValueDataCollection *>(partialResult->get(outputOfStep2ForStep3).get());
    Result                 *results         = static_cast<Result *>(partialResult->get(finalResultFromStep2Master).get());

    size_t na = nBlocks;
    NumericTable **a = new NumericTable*[nBlocks];
    size_t nr = nBlocks + 1;
    NumericTable **r = new NumericTable*[nBlocks + 1];

    r[0] = static_cast<NumericTable *>(results->get(matrixR).get());

    size_t iBlocks = 0;
    for( size_t i = 0; i < nNodes; i++ )
    {
        data_management::DataCollection *nodeCollection = static_cast<data_management::DataCollection *>((*inCollection   ).getValueByIndex(
                                                                                                             i).get());
        data_management::DataCollection *nodePartials   = static_cast<data_management::DataCollection *>((*perNodePartials).getValueByIndex(
                                                                                                             i).get());

        size_t nodeSize = nodeCollection->size();

        for( size_t j = 0 ; j < nodeSize ; j++ )
        {
            a[iBlocks + j    ] = static_cast<NumericTable *>((*nodeCollection)[j].get());
            r[iBlocks + j + 1] = static_cast<NumericTable *>((*nodePartials  )[j].get());
        }

        iBlocks += nodeSize;
    }

    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::QRDistributedStep2Kernel, __DAAL_KERNEL_ARGUMENTS(interm, method), compute, na, a, nr, r, par);

    inCollection->clear();

    delete[] a;
    delete[] r;
}

template<typename interm, Method method, CpuType cpu>
void DistributedContainer<step2Master, interm, method, cpu>::finalizeCompute() {}

template<typename interm, Method method, CpuType cpu>
DistributedContainer<step3Local, interm, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::QRDistributedStep3Kernel, interm, method);
}

template<typename interm, Method method, CpuType cpu>
DistributedContainer<step3Local, interm, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename interm, Method method, CpuType cpu>
void DistributedContainer<step3Local, interm, method, cpu>::compute()
{
    DistributedStep3Input *input = static_cast<DistributedStep3Input *>(_in);
    DistributedPartialResultStep3 *partialResult = static_cast<DistributedPartialResultStep3 *>(_pres);

    data_management::DataCollectionPtr qCollection = input->get(inputOfStep3FromStep1);
    data_management::DataCollectionPtr rCollection = input->get(inputOfStep3FromStep2);

    services::SharedPtr<Result> result = partialResult->get(finalResultFromStep3);

    size_t nBlocks = qCollection->size();

    size_t na = nBlocks * 2;
    NumericTable **a = new NumericTable*[na];
    for(size_t i = 0; i < nBlocks; i++)
    {
        a[i          ] = static_cast<NumericTable *>((*qCollection)[i].get());
        a[i + nBlocks] = static_cast<NumericTable *>((*rCollection)[i].get());
    }

    size_t nr = 1;
    NumericTable *r[1];
    r[0] = static_cast<NumericTable *>(result->get(matrixQ).get());

    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::QRDistributedStep3Kernel, __DAAL_KERNEL_ARGUMENTS(interm, method), compute, na, a, nr, r, par);

    delete[] a;
}

template<typename interm, Method method, CpuType cpu>
void DistributedContainer<step3Local, interm, method, cpu>::finalizeCompute() {}

}
}
} // namespace daal
