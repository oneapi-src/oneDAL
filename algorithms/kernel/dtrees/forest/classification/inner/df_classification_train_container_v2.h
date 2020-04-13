/* file: df_classification_train_container_v2.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Implementation of decision forest container.
//--
*/

#ifndef __DF_CLASSIFICATION_TRAIN_CONTAINER_V2_H__
#define __DF_CLASSIFICATION_TRAIN_CONTAINER_V2_H__

#include "algorithms/kernel/kernel.h"
#include "algorithms/decision_forest/decision_forest_classification_training_types.h"
#include "algorithms/decision_forest/decision_forest_classification_training_batch.h"
#include "algorithms/kernel/dtrees/forest/classification/df_classification_train_kernel.h"
#include "algorithms/kernel/dtrees/forest/classification/df_classification_train_dense_default_kernel.h"
#include "algorithms/kernel/dtrees/forest/classification/oneapi/df_classification_train_hist_kernel_oneapi.h"
#include "algorithms/kernel/dtrees/forest/classification/df_classification_model_impl.h"
#include "service/kernel/service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace training
{
namespace interface2
{
DAAL_FORCEINLINE void convertParameter(interface2::Parameter & par2, interface3::Parameter & par3)
{
    par3.nTrees                      = par2.nTrees;
    par3.observationsPerTreeFraction = par2.observationsPerTreeFraction;
    par3.featuresPerNode             = par2.featuresPerNode;
    par3.maxTreeDepth                = par2.maxTreeDepth;
    par3.minObservationsInLeafNode   = par2.minObservationsInLeafNode;
    par3.seed                        = par2.seed;
    par3.engine                      = par2.engine;
    par3.impurityThreshold           = par2.impurityThreshold;
    par3.varImportance               = par2.varImportance;
    par3.resultsToCompute            = par2.resultsToCompute;
    par3.memorySavingMode            = par2.memorySavingMode;
    par3.bootstrap                   = par2.bootstrap;
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & deviceInfo = context.getInfoDevice();

    if (method == hist && !deviceInfo.isCpu)
    {
        __DAAL_INITIALIZE_KERNELS_SYCL(internal::ClassificationTrainBatchKernelOneAPI, algorithmFPType, method);
    }
    else
    {
        __DAAL_INITIALIZE_KERNELS(internal::ClassificationTrainBatchKernel, algorithmFPType, method);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & deviceInfo = context.getInfoDevice();

    classifier::training::Input * input = static_cast<classifier::training::Input *>(_in);
    Result * result                     = static_cast<Result *>(_res);

    NumericTable * x = input->get(classifier::training::data).get();
    NumericTable * y = input->get(classifier::training::labels).get();

    decision_forest::classification::Model * m = result->get(classifier::training::model).get();

    m->setNFeatures(x->getNumberOfColumns());

    decision_forest::classification::training::interface2::Parameter * par =
        static_cast<decision_forest::classification::training::interface2::Parameter *>(_par);
    decision_forest::classification::training::interface3::Parameter par3(par->nClasses);
    convertParameter(*par, par3);
    daal::services::Environment::env & env = *_env;

    if (method == hist && !deviceInfo.isCpu)
    {
        __DAAL_CALL_KERNEL_SYCL(env, internal::ClassificationTrainBatchKernelOneAPI, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                                daal::services::internal::hostApp(*input), x, y, *m, *result, par3);
    }
    else
    {
        __DAAL_CALL_KERNEL(env, internal::ClassificationTrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                           daal::services::internal::hostApp(*input), x, y, *m, *result, par3);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    Result * result                                              = static_cast<Result *>(_res);
    decision_forest::classification::Model * m                   = result->get(classifier::training::model).get();
    decision_forest::classification::internal::ModelImpl * pImpl = dynamic_cast<decision_forest::classification::internal::ModelImpl *>(m);
    DAAL_ASSERT(pImpl);
    pImpl->clear();
    return services::Status();
}

} // namespace interface2

} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
