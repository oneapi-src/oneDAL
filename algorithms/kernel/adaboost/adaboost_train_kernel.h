/* file: adaboost_train_kernel.h */
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
//  Declaration of template function that trains Ada Boost model.
//--
*/

#ifndef __ADABOOST_TRAIN_KERNEL_H__
#define __ADABOOST_TRAIN_KERNEL_H__

#include "adaboost_model.h"
#include "adaboost_training_types.h"
#include "kernel.h"
#include "service_numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace training
{
namespace internal
{
template <Method method, typename algorithmFPType, CpuType cpu>
class AdaBoostTrainKernel : public Kernel
{
public:
    services::Status compute(NumericTablePtr * a, Model * r, NumericTable * weakLearnersErrorsTable, const Parameter * par);
    typedef typename daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    typedef typename services::SharedPtr<HomogenNT> HomogenNTPtr;

private:
    services::Status adaboostSAMME(size_t nVectors, NumericTablePtr weakLearnerInputTables[], const algorithmFPType * y, Model * boostModel,
                                   algorithmFPType * weakLearnersErrorsTable, const Parameter * parameter, size_t & nWeakLearners,
                                   algorithmFPType * alpha);
    services::Status adaboostSAMME_R(size_t nVectors, NumericTablePtr weakLearnerInputTables[], const algorithmFPType * y, Model * boostModel,
                                     algorithmFPType * weakLearnersErrorsTable, const Parameter * parameter, size_t & nWeakLearners,
                                     algorithmFPType * alpha);
    void convertLabelToVector(size_t nClasses, algorithmFPType * Y);
};
} // namespace internal
} // namespace training
} // namespace adaboost
} // namespace algorithms
} // namespace daal

#endif
