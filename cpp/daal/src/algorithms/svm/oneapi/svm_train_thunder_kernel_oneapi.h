/* file: svm_train_thunder_kernel_oneapi.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

//++
//  Declaration of template structs that calculate SVM Training functions.
//--

#ifndef __SVM_TRAIN_THUNDER_KERNEL_ONEAPI_H__
#define __SVM_TRAIN_THUNDER_KERNEL_ONEAPI_H__

#include "services/env_detect.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/svm/svm_train_types.h"
#include "src/algorithms/kernel.h"
#include "src/algorithms/svm/oneapi/svm_helper_oneapi.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::services;

template <typename algorithmFPType, Method method>
class SVMTrainOneAPI : public Kernel
{
public:
    services::Status compute(const NumericTablePtr & xTable, NumericTable & yTable, daal::algorithms::Model * r, const ParameterType * par)
    {
        return services::ErrorMethodNotImplemented;
    }
};

template <typename algorithmFPType>
class SVMTrainOneAPI<algorithmFPType, thunder> : public Kernel
{
    using Helper = utils::internal::HelperSVM<algorithmFPType>;

public:
    services::Status compute(const NumericTablePtr & xTable, NumericTable & yTable, daal::algorithms::Model * r, const ParameterType * par);

protected:
    services::Status updateGrad(const services::Buffer<algorithmFPType> & kernelWS, const services::Buffer<algorithmFPType> & deltaalpha,
                                services::Buffer<algorithmFPType> & grad, const size_t nVectors, const size_t nWS);
    services::Status smoKernel(const services::Buffer<algorithmFPType> & y, const services::Buffer<algorithmFPType> & kernelWsRows,
                               const services::Buffer<uint32_t> & wsIndices, const uint32_t ldK, const services::Buffer<algorithmFPType> & f,
                               const algorithmFPType C, const algorithmFPType accuracyThreshold, const algorithmFPType tau,
                               const uint32_t maxInnerIteration, services::Buffer<algorithmFPType> & alpha,
                               services::Buffer<algorithmFPType> & deltaalpha, services::Buffer<algorithmFPType> & resinfo, const size_t nWS);

    bool checkStopCondition(const algorithmFPType diff, const algorithmFPType diffPrev, const algorithmFPType accuracyThreshold,
                            size_t & sameLocalDiff);

private:
    // One of the conditions for stopping is diff stays unchanged. nNoChanges - number of repetitions
    static const size_t nNoChanges = 5;
    // The maximum numbers of iteration of the subtask is number of observation in WS x cInnerIterations. It's enough to find minimum for subtask.
    static const size_t cInnerIterations = 1000;
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
