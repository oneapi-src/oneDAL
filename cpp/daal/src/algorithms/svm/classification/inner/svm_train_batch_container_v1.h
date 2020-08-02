/* file: svm_train_batch_container_v1.h */
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
//  Implementation of SVM training algorithm container.
//--
*/

#include "algorithms/svm/svm_train.h"
#include "src/algorithms/svm/svm_train_kernel.h"
#include "src/algorithms/svm/svm_train_boser_kernel.h"
#include "algorithms/classifier/classifier_training_types.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace interface1
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;

/**
*  \brief Initialize list of SVM kernels with implementations for supported architectures
*/
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SVMTrainImpl, method, svm::interface1::Parameter, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::training::Input * input = static_cast<classifier::training::Input *>(_in);
    svm::training::Result * result      = static_cast<svm::training::Result *>(_res);

    const NumericTablePtr x = input->get(classifier::training::data);
    const NumericTablePtr y = input->get(classifier::training::labels);
    const NumericTablePtr weights;

    algorithms::Model * r = static_cast<daal::algorithms::Model *>(result->get(classifier::training::model).get());

    const svm::interface1::Parameter * const par1 = static_cast<svm::interface1::Parameter *>(_par);
    svm::interface2::Parameter par2;

    par2.nClasses          = par1->nClasses;
    par2.C                 = par1->C;
    par2.accuracyThreshold = par1->accuracyThreshold;
    par2.tau               = par1->tau;
    par2.maxIterations     = par1->maxIterations;
    par2.kernel            = par1->kernel;
    par2.shrinkingStep     = par1->shrinkingStep;
    par2.doShrinking       = par1->doShrinking;
    par2.cacheSize         = par1->cacheSize;

    services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::SVMTrainImpl, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, x, weights,
                       *y, r, &par2);
}
} // namespace interface1
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal
