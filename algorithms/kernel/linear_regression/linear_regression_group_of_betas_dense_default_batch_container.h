/* file: linear_regression_group_of_betas_dense_default_batch_container.h */
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
//  Implementation of the container for the multi-class confusion matrix.
//--
*/

#ifndef __LINEAR_REGRESSION_GROUP_OF_BETAS_DENSE_DEFAULT_BATCH_CONTAINER_H__
#define __LINEAR_REGRESSION_GROUP_OF_BETAS_DENSE_DEFAULT_BATCH_CONTAINER_H__

#include "algorithms/linear_regression/linear_regression_group_of_betas_batch.h"
#include "linear_regression_group_of_betas_dense_default_batch_kernel.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace quality_metric
{
namespace group_of_betas
{

using namespace daal::data_management;

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::GroupOfBetasKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input* input = static_cast<Input* >(_in);
    Result* result = static_cast<Result* >(_res);
    Parameter* par = static_cast<Parameter* >(_par);

    NumericTable* out[] = {
        result->get(expectedMeans).get(),
        result->get(expectedVariance).get(),
        result->get(regSS).get(),
        result->get(resSS).get(),
        result->get(tSS).get(),
        result->get(determinationCoeff).get(),
        result->get(fStatistics).get()
    };

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::GroupOfBetasKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType),
        compute, input->get(expectedResponses).get(),
        input->get(predictedResponses).get(),
        input->get(predictedReducedModelResponses).get(),
        par->numBeta, par->numBetaReducedModel, par->accuracyThreshold, out);
}

}
}
}
}
}

#endif
