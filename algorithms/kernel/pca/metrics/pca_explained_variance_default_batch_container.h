/* file: pca_explained_variance_default_batch_container.h */
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

#ifndef __PCA_EXPLAINED_VARIANCE_DEFAULT_BATCH_CONTAINER_H__
#define __PCA_EXPLAINED_VARIANCE_DEFAULT_BATCH_CONTAINER_H__

#include "algorithms/pca/pca_explained_variance_batch.h"
#include "pca_explained_variance_default_batch_kernel.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace quality_metric
{
namespace explained_variance
{

using namespace daal::data_management;

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::ExplainedVarianceKernel, method, algorithmFPType);
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

    auto& explainedVariancesTable =       *(result->get(explainedVariances));
    auto& explainedVariancesRatiosTable = *(result->get(explainedVariancesRatios));
    auto& noiseVarianceTable =            *(result->get(noiseVariance));

    const auto& eigenvaluesTable = *(input->get(eigenvalues));

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::ExplainedVarianceKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType),
        compute, eigenvaluesTable,
        explainedVariancesTable,
        explainedVariancesRatiosTable,
        noiseVarianceTable);
}

}
}
}
}
}

#endif
