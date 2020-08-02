/* file: svm_train_thunder_kernel.h */
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

/*
//++
//  Declaration of template structs that calculate SVM Training functions.
//--
*/

#ifndef __SVM_TRAIN_THUNDER_KERNEL_H__
#define __SVM_TRAIN_THUNDER_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/model.h"
#include "services/daal_defines.h"
#include "algorithms/svm/svm_train_types.h"
#include "src/algorithms/kernel.h"
#include "src/data_management/service_micro_table.h"

#include "src/algorithms/svm/svm_train_kernel.h"

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
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
struct SVMTrainImpl<thunder, algorithmFPType, ParameterType, cpu> : public Kernel
{
    services::Status compute(const data_management::NumericTablePtr & xTable, const data_management::NumericTablePtr & wTable,
                             data_management::NumericTable & yTable, daal::algorithms::Model * r, const ParameterType * par);

private:
    services::Status SMOBlockSolver(const algorithmFPType * y, const algorithmFPType * grad, const uint32_t * wsIndices,
                                    const NumericTablePtr & kernelWS, const size_t nVectors, const size_t nWS, const algorithmFPType * cw,
                                    const double accuracyThreshold, const double tau, algorithmFPType * buffer, char * I, algorithmFPType * alpha,
                                    algorithmFPType * deltaAlpha, algorithmFPType & localDiff) const;

    services::Status updateGrad(const NumericTablePtr & kernelWS, const algorithmFPType * deltaalpha, algorithmFPType * tmpgrad,
                                algorithmFPType * grad, const size_t nVectors, const size_t nWS);

    bool checkStopCondition(const algorithmFPType diff, const algorithmFPType diffPrev, const algorithmFPType accuracyThreshold,
                            size_t & sameLocalDiff);
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
