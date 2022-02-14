/* file: svm_predict_kernel_oneapi.h */
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
//  Declaration of template structs that contains SVM prediction functions.
//--
*/

#ifndef __SVM_PREDICT_KERNEL_ONEAPI_H__
#define __SVM_PREDICT_KERNEL_ONEAPI_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/model.h"
#include "algorithms/svm/svm_predict_types.h"
#include "src/algorithms/kernel.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace prediction
{
namespace internal
{
using namespace daal::data_management;

template <Method method, typename algorithmFPType>
struct SVMPredictImplOneAPI : public Kernel
{
    services::Status compute(const NumericTablePtr & xTable, Model * model, NumericTable & r, const svm::Parameter * par)
    {
        return services::ErrorMethodNotImplemented;
    }
};

template <typename algorithmFPType>
struct SVMPredictImplOneAPI<defaultDense, algorithmFPType> : public Kernel
{
    services::Status compute(const NumericTablePtr & xTable, Model * model, NumericTable & r, const svm::Parameter * par);
};

} // namespace internal
} // namespace prediction
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
