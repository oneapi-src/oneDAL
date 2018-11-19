/* file: gbt_regression_train_container.h */
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
//  Implementation of gradient boosted trees container.
//--
*/

#ifndef __GBT_REGRESSION_TRAIN_CONTAINER_H__
#define __GBT_REGRESSION_TRAIN_CONTAINER_H__

#include "kernel.h"
#include "gbt_regression_training_types.h"
#include "gbt_regression_training_batch.h"
#include "gbt_regression_train_kernel.h"
#include "gbt_regression_model_impl.h"
#include "service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace training
{

/**
 *  \brief Initialize list of gradient boosted trees
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::RegressionTrainBatchKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate gradient boosted trees model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Decision forest algorithm parameters
 */
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    const NumericTable *x = input->get(data).get();
    const NumericTable *y = input->get(dependentVariable).get();

    gbt::regression::Model *m = result->get(model).get();

    const Parameter *par = static_cast<gbt::regression::training::Parameter*>(_par);
    daal::services::Environment::env &env = *_env;
    daal::algorithms::engines::internal::BatchBaseImpl* engine = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(par->engine.get());

    __DAAL_CALL_KERNEL(env, internal::RegressionTrainBatchKernel,
        __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, daal::services::internal::hostApp(*input), x, y, *m, *result, *par, *engine);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    Result *result = static_cast<Result *>(_res);
    gbt::regression::Model *m = result->get(model).get();
    gbt::regression::internal::ModelImpl* pImpl = dynamic_cast<gbt::regression::internal::ModelImpl*>(m);
    DAAL_ASSERT(pImpl);
    pImpl->clear();
    return services::Status();
}

}
}
}
}
}
#endif
