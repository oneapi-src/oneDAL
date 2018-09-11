/* file: implicit_als_predict_ratings_dense_default_container.h */
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
//++
//  Implementation of implicit ALS prediction algorithm container.
//--
*/

#ifndef __IMPLICIT_ALS_PREDICT_RATINGS_DENSE_DEFAULT_CONTAINER_H__
#define __IMPLICIT_ALS_PREDICT_RATINGS_DENSE_DEFAULT_CONTAINER_H__

#include "implicit_als_predict_ratings_batch.h"
#include "implicit_als_predict_ratings_dense_default_kernel.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace prediction
{
namespace ratings
{
/**
 *  \brief Initialize list of implicit ALS prediction algorithm
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, prediction::ratings::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv) : PredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSPredictKernel, algorithmFPType);
}

template <typename algorithmFPType, prediction::ratings::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, prediction::ratings::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    Model        *alsModel     = static_cast<Model *>(input->get(model)  .get());

    NumericTable *usersFactorsTable = alsModel->getUsersFactors().get();
    NumericTable *itemsFactorsTable = alsModel->getItemsFactors().get();

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    NumericTable *ratingsTable = static_cast<NumericTable *>(result->get(prediction).get());
    __DAAL_CALL_KERNEL(env, internal::ImplicitALSPredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType),
                       compute, usersFactorsTable, itemsFactorsTable, ratingsTable, par);
}

/**
 *  \brief Initialize list of implicit ALS prediction algorithm
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, prediction::ratings::Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv) :
    DistributedPredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSPredictKernel, algorithmFPType);
}

template <typename algorithmFPType, prediction::ratings::Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, prediction::ratings::Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step1Local> *input = static_cast<DistributedInput<step1Local> *>(_in);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);
    Result *result = static_cast<Result *>(partialResult->get(finalResult).get());

    PartialModel *usersFactors = static_cast<PartialModel *>(input->get(usersPartialModel).get());
    PartialModel *itemsFactors = static_cast<PartialModel *>(input->get(itemsPartialModel).get());

    NumericTablePtr usersFactorsTable = usersFactors->getFactors();
    NumericTablePtr itemsFactorsTable = itemsFactors->getFactors();

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    NumericTable *ratingsTable = static_cast<NumericTable *>(result->get(prediction).get());

    __DAAL_CALL_KERNEL(env, internal::ImplicitALSPredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType),
                       compute, usersFactorsTable.get(), itemsFactorsTable.get(), ratingsTable, par);
}

template <typename algorithmFPType, prediction::ratings::Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

} // ratings
} // prediction
} // implicit_als
} // algorithms
} // daal

#endif
