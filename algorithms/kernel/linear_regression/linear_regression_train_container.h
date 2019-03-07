/* file: linear_regression_train_container.h */
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
//  Implementation of linear regression container.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_CONTAINER_H__
#define __LINEAR_REGRESSION_TRAIN_CONTAINER_H__

#include "kernel.h"
#include "linear_regression_training_batch.h"
#include "linear_regression_training_online.h"
#include "linear_regression_training_distributed.h"
#include "linear_regression_train_kernel.h"
#include "linear_regression_ne_model.h"
#include "linear_regression_qr_model.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
using namespace daal::data_management;
using namespace daal::services;
using namespace daal::internal;

/**
 *  \brief Initialize list of linear regression
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::BatchKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate linear regression model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Linear regression algorithm parameters
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);
    Parameter *par = static_cast<Parameter *>(_par);

    Environment::env &env = *_env;

    if (method == training::normEqDense)
    {
        linear_regression::ModelNormEqPtr m = linear_regression::ModelNormEq::cast(result->get(model));

        __DAAL_CALL_KERNEL(env, internal::BatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::normEqDense), \
                           compute, *(input->get(data)), *(input->get(dependentVariables)),                             \
                           *(m->getXTXTable()), *(m->getXTYTable()), *(m->getBeta()), par->interceptFlag);
    }
    else
    {
        linear_regression::ModelQRPtr m = linear_regression::ModelQR::cast(result->get(model));

        __DAAL_CALL_KERNEL(env, internal::BatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::qrDense),    \
                           compute, *(input->get(data)), *(input->get(dependentVariables)),                            \
                           *(m->getRTable()), *(m->getQTYTable()), *(m->getBeta()), par->interceptFlag);
    }
}


/**
 *  \brief Initialize list of linear regression
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::OnlineContainer(Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::OnlineKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::~OnlineContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate linear regression model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Linear regression algorithm parameters
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
Status OnlineContainer<algorithmFPType, method, cpu>::compute()
{
    linear_regression::training::Input *input = static_cast<linear_regression::training::Input *>(_in);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);
    Parameter *par = static_cast<Parameter *>(_par);

    Environment::env &env = *_env;

    if (method == training::normEqDense)
    {
        linear_regression::ModelNormEqPtr m = linear_regression::ModelNormEq::cast(
                partialResult->get(training::partialModel));
        __DAAL_CALL_KERNEL(env, internal::OnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::normEqDense),\
                       compute, *(input->get(data)), *(input->get(dependentVariables)),                                 \
                       *(m->getXTXTable()), *(m->getXTYTable()), par->interceptFlag);
    }
    else
    {
        linear_regression::ModelQRPtr m = linear_regression::ModelQR::cast(
                partialResult->get(training::partialModel));

        __DAAL_CALL_KERNEL(env, internal::OnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::qrDense),    \
                       compute, *(input->get(data)), *(input->get(dependentVariables)),                                 \
                       *(m->getRTable()), *(m->getQTYTable()), par->interceptFlag);
    }
}

/**
 *  \brief Choose appropriate kernel to calculate linear regression model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Linear regression algorithm parameters
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
Status OnlineContainer<algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);
    Result *result = static_cast<Result *>(_res);
    Parameter *par = static_cast<Parameter *>(_par);
    Environment::env &env = *_env;

    if (method == training::normEqDense)
    {
        linear_regression::ModelNormEqPtr pm = linear_regression::ModelNormEq::cast(
                partialResult->get(training::partialModel));
        linear_regression::ModelNormEqPtr m = linear_regression::ModelNormEq::cast(
                result->get(training::model));
        __DAAL_CALL_KERNEL(env, internal::OnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::normEqDense),   \
                           finalizeCompute, *(pm->getXTXTable()), *(pm->getXTYTable()),                                    \
                           *(m->getXTXTable()), *(m->getXTYTable()), *(m->getBeta()), par->interceptFlag);
    }
    else
    {
        linear_regression::ModelQRPtr pm = linear_regression::ModelQR::cast(
                partialResult->get(training::partialModel));
        linear_regression::ModelQRPtr m = linear_regression::ModelQR::cast(
                result->get(training::model));
        __DAAL_CALL_KERNEL(env, internal::OnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::qrDense),   \
                           finalizeCompute, *(pm->getRTable()), *(pm->getQTYTable()),                                  \
                           *(m->getRTable()), *(m->getQTYTable()), *(m->getBeta()), par->interceptFlag);
    }
}


/**
 *  \brief Initialize list of linear regression
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::DistributedKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate linear regression model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Linear regression algorithm parameters
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step2Master> *input = static_cast<DistributedInput<step2Master> *>(_in);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);

    Environment::env &env = *_env;

    DataCollectionPtr collection = input->get(partialModels);
    size_t n = collection->size();
    Status s;
    if (method == training::normEqDense)
    {
        TArray<NumericTable *, sse2> partialxtx(n);
        TArray<NumericTable *, sse2> partialxty(n);
        for(size_t i = 0; i < n; i++)
        {
            linear_regression::ModelNormEq *m = static_cast<linear_regression::ModelNormEq *>(
                (*collection)[i].get());
            partialxtx[i] = m->getXTXTable().get();
            partialxty[i] = m->getXTYTable().get();
        }

        linear_regression::ModelNormEqPtr pm = linear_regression::ModelNormEq::cast(
                partialResult->get(training::partialModel));

        s = __DAAL_CALL_KERNEL_STATUS(env, internal::DistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::normEqDense),  \
                           compute, n, partialxtx.get(), partialxty.get(),  \
                           *(pm->getXTXTable()), *(pm->getXTYTable()));
    }
    else
    {
        TArray<NumericTable *, sse2> partialr(n);
        TArray<NumericTable *, sse2> partialqty(n);
        for(size_t i = 0; i < n; i++)
        {
            linear_regression::ModelQR *m = static_cast<linear_regression::ModelQR *>(
                (*collection)[i].get());
            partialr[i] = m->getRTable().get();
            partialqty[i] = m->getQTYTable().get();
        }

        linear_regression::ModelQRPtr pm = linear_regression::ModelQR::cast(
                partialResult->get(training::partialModel));

        s = __DAAL_CALL_KERNEL_STATUS(env, internal::DistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::qrDense),  \
                           compute, n, partialr.get(), partialqty.get(),  \
                           *(pm->getRTable()), *(pm->getQTYTable()));
    }
    collection->clear();
    return s;
}

/**
 *  \brief Choose appropriate kernel to calculate linear regression model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Linear regression algorithm parameters
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);
    Result *result = static_cast<Result *>(_res);
    Parameter *par = static_cast<Parameter *>(_par);

    Environment::env &env = *_env;

    if (method == training::normEqDense)
    {
        linear_regression::ModelNormEqPtr pm = linear_regression::ModelNormEq::cast(
                partialResult->get(training::partialModel));
        linear_regression::ModelNormEqPtr m  = linear_regression::ModelNormEq::cast(
                result->get(training::model));
        __DAAL_CALL_KERNEL(env, internal::DistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::normEqDense),   \
                           finalizeCompute, *(pm->getXTXTable()), *(pm->getXTYTable()),                                    \
                           *(m->getXTXTable()), *(m->getXTYTable()), *(m->getBeta()), par->interceptFlag);
    }
    else
    {
        linear_regression::ModelQRPtr pm = linear_regression::ModelQR::cast(
                partialResult->get(training::partialModel));
        linear_regression::ModelQRPtr m  = linear_regression::ModelQR::cast(
                result->get(training::model));

        __DAAL_CALL_KERNEL(env, internal::DistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::qrDense),  \
                           finalizeCompute, *(pm->getRTable()), *(pm->getQTYTable()),                                    \
                           *(m->getRTable()), *(m->getQTYTable()), *(m->getBeta()), par->interceptFlag);
    }
}

}
}
}
}

#endif
