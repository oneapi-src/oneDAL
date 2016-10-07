/* file: adaboost_train_kernel.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
    void compute(size_t n, NumericTablePtr *a, Model *r, const Parameter *par);

private:
    void adaBoostFreundKernel(size_t nVectors, NumericTablePtr weakLearnerInputTables[],
                              services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > hTable, algorithmFPType *y,
                              Model *boostModel, Parameter *parameter, size_t *nWeakLearnersPtr,
                              algorithmFPType **alphaPtr);
};

} // namespace daal::algorithms::adaboost::training::internal
}
}
}
} // namespace daal

#endif
