/* file: gbt_regression_train_container.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of gradient boosted trees container.
//--
*/

#ifndef __GBT_REGRESSION_TRAIN_CONTAINER_H__
#define __GBT_REGRESSION_TRAIN_CONTAINER_H__

#include "kernel.h"
#include "gbt_regression_training_types.h"
#include "gbt_regression_training_batch.h"
#include "gbt_regression_training_distributed.h"
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
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
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
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    const NumericTable * x = input->get(data).get();
    const NumericTable * y = input->get(dependentVariable).get();

    gbt::regression::Model * m = result->get(model).get();

    const Parameter * par                  = static_cast<gbt::regression::training::Parameter *>(_par);
    daal::services::Environment::env & env = *_env;
    daal::algorithms::engines::internal::BatchBaseImpl * engine =
        dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl *>(par->engine.get());

    __DAAL_CALL_KERNEL(env, internal::RegressionTrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       daal::services::internal::hostApp(*input), x, y, *m, *result, *par, *engine);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    Result * result                              = static_cast<Result *>(_res);
    gbt::regression::Model * m                   = result->get(model).get();
    gbt::regression::internal::ModelImpl * pImpl = dynamic_cast<gbt::regression::internal::ModelImpl *>(m);
    DAAL_ASSERT(pImpl);
    pImpl->clear();
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv) :
    TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::RegressionTrainDistrStep1Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step1Local> *input = static_cast<DistributedInput<step1Local> *>(_in);
    DistributedPartialResultStep1 *partialResult = static_cast<DistributedPartialResultStep1 *>(_pres);

    const NumericTablePtr ntBinnedData         = input->get(step1BinnedData);
    const NumericTablePtr ntDependentVariable  = input->get(step1DependentVariable);
    const NumericTablePtr ntInputResponse      = input->get(step1InputResponse);
    const NumericTablePtr ntInputTreeStructure = input->get(step1InputTreeStructure);
    const NumericTablePtr ntInputTreeOrder     = input->get(step1InputTreeOrder);

    const NumericTablePtr ntResponse      = partialResult->get(response);
    const NumericTablePtr ntOptCoeffs     = partialResult->get(optCoeffs);
    const NumericTablePtr ntTreeOrder     = partialResult->get(treeOrder);
    const NumericTablePtr ntFinalizedTree = partialResult->get(finalizedTree);
    const NumericTablePtr ntTreeStructure = partialResult->get(step1TreeStructure);

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::RegressionTrainDistrStep1Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, ntBinnedData.get(), ntDependentVariable.get(), ntInputResponse.get(),
                       ntInputTreeStructure.get(), ntInputTreeOrder.get(),
                       ntResponse.get(), ntOptCoeffs.get(), ntTreeOrder.get(), ntFinalizedTree.get(), ntTreeStructure.get(), *par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}


template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv) :
    TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::RegressionTrainDistrStep2Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step2Local> *input = static_cast<DistributedInput<step2Local> *>(_in);
    DistributedPartialResultStep2 *partialResult = static_cast<DistributedPartialResultStep2 *>(_pres);

    const NumericTablePtr ntInputTreeStructure = input->get(step2InputTreeStructure);

    const NumericTablePtr ntFinishedFlag  = partialResult->get(finishedFlag);

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::RegressionTrainDistrStep2Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, ntInputTreeStructure.get(), ntFinishedFlag.get());
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}


template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step3Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv) :
    TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::RegressionTrainDistrStep3Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step3Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step3Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step3Local> *input = static_cast<DistributedInput<step3Local> *>(_in);
    DistributedPartialResultStep3 *partialResult = static_cast<DistributedPartialResultStep3 *>(_pres);

    const NumericTablePtr   ntBinnedData         = input->get(step3BinnedData);
    const NumericTablePtr   ntBinSizes           = input->get(step3BinSizes);
    const NumericTablePtr   ntInputTreeStructure = input->get(step3InputTreeStructure);
    const NumericTablePtr   ntInputTreeOrder     = input->get(step3InputTreeOrder);
    const NumericTablePtr   ntOptCoeffs          = input->get(step3OptCoeffs);
    const DataCollectionPtr dcParentHistograms   = input->get(step3ParentHistograms);

    const DataCollectionPtr dcHistograms = partialResult->get(histograms);

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::RegressionTrainDistrStep3Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, ntBinnedData.get(), ntBinSizes.get(), ntInputTreeStructure.get(), ntInputTreeOrder.get(),
                       ntOptCoeffs.get(), dcParentHistograms.get(), dcHistograms.get());
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step3Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}


template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step4Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv) :
    TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::RegressionTrainDistrStep4Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step4Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step4Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step4Local> *input = static_cast<DistributedInput<step4Local> *>(_in);
    DistributedPartialResultStep4 *partialResult = static_cast<DistributedPartialResultStep4 *>(_pres);

    const NumericTablePtr   ntInputTreeStructure               = input->get(step4InputTreeStructure);
    const DataCollectionPtr dcParentTotalHistogramsForFeatures = input->get(step4ParentTotalHistograms);
    const DataCollectionPtr dcPartialHistogramsForFeatures     = input->get(step4PartialHistograms);
    const DataCollectionPtr dcFeatureIndices                   = input->get(step4FeatureIndices);

    const DataCollectionPtr dcTotalHistogramsForFeatures = partialResult->get(totalHistograms);
    const DataCollectionPtr dcBestSplitsForFeatures      = partialResult->get(bestSplits);

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::RegressionTrainDistrStep4Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, ntInputTreeStructure.get(), dcParentTotalHistogramsForFeatures.get(), dcPartialHistogramsForFeatures.get(), dcFeatureIndices.get(),
                       dcTotalHistogramsForFeatures.get(), dcBestSplitsForFeatures.get(), *par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step4Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}


template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step5Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv) :
    TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::RegressionTrainDistrStep5Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step5Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step5Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step5Local> *input = static_cast<DistributedInput<step5Local> *>(_in);
    DistributedPartialResultStep5 *partialResult = static_cast<DistributedPartialResultStep5 *>(_pres);

    const NumericTablePtr   ntBinnedData         = input->get(step5BinnedData);
    const NumericTablePtr   ntTransposedBinnedData = input->get(step5TransposedBinnedData);
    const NumericTablePtr   ntBinSizes           = input->get(step5BinSizes);
    const NumericTablePtr   ntInputTreeStructure = input->get(step5InputTreeStructure);
    const NumericTablePtr   ntInputTreeOrder     = input->get(step5InputTreeOrder);
    const DataCollectionPtr dcPartialBestSplits  = input->get(step5PartialBestSplits);

    const NumericTablePtr ntTreeStructure = partialResult->get(step5TreeStructure);
    const NumericTablePtr ntTreeOrder     = partialResult->get(step5TreeOrder);

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::RegressionTrainDistrStep5Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, ntBinnedData.get(), ntTransposedBinnedData.get(), ntBinSizes.get(), ntInputTreeStructure.get(), ntInputTreeOrder.get(),
                       dcPartialBestSplits.get(), ntTreeStructure.get(), ntTreeOrder.get(), *par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step5Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}


template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step6Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv) :
    TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::RegressionTrainDistrStep6Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step6Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step6Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step6Local> *input = static_cast<DistributedInput<step6Local> *>(_in);
    DistributedPartialResultStep6 *partialResult = static_cast<DistributedPartialResultStep6 *>(_pres);

    const NumericTablePtr   ntInitialResponse = input->get(step6InitialResponse);
    const DataCollectionPtr dcBinValues       = input->get(step6BinValues);
    const DataCollectionPtr dcFinalizedTrees  = input->get(step6FinalizedTrees);

    const gbt::regression::ModelPtr model = partialResult->get(partialModel);

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::RegressionTrainDistrStep6Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, ntInitialResponse.get(), dcBinValues.get(), dcFinalizedTrees.get(), model.get(), *par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step6Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

}
}
}
}
}
#endif
