/* file: linear_regression_single_beta_dense_default_batch_container.h */
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
//  Implementation of the container for the multi-class confusion matrix.
//--
*/

#ifndef __LINEAR_REGRESSION_SINGLE_BETA_DENSE_DEFAULT_BATCH_CONTAINER_H__
#define __LINEAR_REGRESSION_SINGLE_BETA_DENSE_DEFAULT_BATCH_CONTAINER_H__

#include "algorithms/linear_regression/linear_regression_single_beta_batch.h"
#include "linear_regression_single_beta_dense_default_batch_kernel.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace quality_metric
{
namespace single_beta
{
using namespace daal::data_management;

namespace internal
{
const NumericTable * getXtXTable(const linear_regression::Model & model, bool & bModelNe);

}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SingleBetaKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);
    Parameter * par = static_cast<Parameter *>(_par);

    data_management::DataCollection * coll = result->get(betaCovariances).get();
    internal::SingleBetaOutput out(coll->size());
    out.rms                 = result->get(rms).get();
    out.variance            = result->get(variance).get();
    out.zScore              = result->get(zScore).get();
    out.confidenceIntervals = result->get(confidenceIntervals).get();
    out.inverseOfXtX        = result->get(inverseOfXtX).get();
    for (size_t i = 0; i < coll->size(); ++i) out.betaCovariances[i] = dynamic_cast<NumericTable *>((*coll)[i].get());

    const auto pModel                      = input->get(model).get();
    bool bModelNe                          = false;
    const auto pXtx                        = internal::getXtXTable(*pModel, bModelNe);
    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::SingleBetaKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute,
                       input->get(expectedResponses).get(), input->get(predictedResponses).get(), pModel->getNumberOfFeatures(),
                       pModel->getBeta().get(), pXtx, bModelNe, par->accuracyThreshold, par->alpha, out);
}

} // namespace single_beta
} // namespace quality_metric
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
