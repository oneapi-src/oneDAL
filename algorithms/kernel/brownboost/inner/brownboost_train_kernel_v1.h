/* file: brownboost_train_kernel_v1.h */
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
//  Declaration of template function that trains Brown Boost model.
//--
*/

#ifndef __BROWN_BOOST_TRAIN_KERNEL_V1_H___
#define __BROWN_BOOST_TRAIN_KERNEL_V1_H___

#include "brownboost_model.h"
#include "brownboost_training_types.h"
#include "kernel.h"
#include "service_numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace training
{
namespace internal
{
template <Method method, typename algorithmFPType, CpuType cpu>
class I1BrownBoostTrainKernel : public Kernel
{
public:
    services::Status compute(size_t n, NumericTablePtr * a, brownboost::interface1::Model * r, const brownboost::interface1::Parameter * par);

private:
    typedef typename daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    typedef typename services::SharedPtr<HomogenNT> HomogenNTPtr;

    void updateWeights(size_t nVectors, algorithmFPType s, algorithmFPType c, algorithmFPType invSqrtC, const algorithmFPType * r,
                       algorithmFPType * nra, algorithmFPType * nre2, algorithmFPType * w);

    algorithmFPType * reallocateAlpha(size_t oldAlphaSize, size_t alphaSize, algorithmFPType * oldAlpha, services::Status & s);

    services::Status brownBoostFreundKernel(size_t nVectors, NumericTablePtr weakLearnerInputTables[], const HomogenNTPtr & hTable,
                                            const algorithmFPType * y, brownboost::interface1::Model * boostModel,
                                            brownboost::interface1::Parameter * parameter, size_t & nWeakLearners, algorithmFPType *& alpha);
};

} // namespace internal
} // namespace training
} // namespace brownboost
} // namespace algorithms
} // namespace daal

#endif
