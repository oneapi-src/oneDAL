/* file: svm_train_batch_container.h */
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
#include "algorithms/kernel/svm/svm_train_kernel.h"
#include "algorithms/kernel/svm/svm_train_boser_kernel.h"
#include "algorithms/classifier/classifier_training_types.h"
#include "algorithms/kernel/svm/oneapi/svm_train_thunder_kernel_oneapi.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace interface2
{
using namespace daal::data_management;

/**
*  \brief Initialize list of SVM kernels with implementations for supported architectures
*/
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & deviceInfo = context.getInfoDevice();
    if (method == thunder && !deviceInfo.isCpu)
    {
        __DAAL_INITIALIZE_KERNELS_SYCL(internal::SVMTrainOneAPI, algorithmFPType, svm::interface2::Parameter, method);
    }
    else
    {
        __DAAL_INITIALIZE_KERNELS(internal::SVMTrainImpl, method, svm::interface2::Parameter, algorithmFPType);
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
    classifier::training::Input * input = static_cast<classifier::training::Input *>(_in);
    svm::training::Result * result      = static_cast<svm::training::Result *>(_res);

    NumericTablePtr x = input->get(classifier::training::data);
    NumericTablePtr y = input->get(classifier::training::labels);

    daal::algorithms::Model * r = static_cast<daal::algorithms::Model *>(result->get(classifier::training::model).get());

    svm::interface2::Parameter * par       = static_cast<svm::interface2::Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & deviceInfo = context.getInfoDevice();
    if (method == thunder && !deviceInfo.isCpu)
    {
        __DAAL_CALL_KERNEL_SYCL(env, internal::SVMTrainOneAPI, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, svm::interface2::Parameter, method), compute,
                                x, *y, r, par);
    }
    else
    {
        __DAAL_CALL_KERNEL(env, internal::SVMTrainImpl, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType, svm::interface2::Parameter), compute, x, *y,
                           r, par);
    }
}
} // namespace interface2
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal
