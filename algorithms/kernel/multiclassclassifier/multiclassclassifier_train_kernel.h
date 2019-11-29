/* file: multiclassclassifier_train_kernel.h */
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
//  Declaration of template function that trains Multi-class slassifier model.
//--
*/

#ifndef __MULTICLASSCLASSIFIER_TRAIN_KERNEL_H__
#define __MULTICLASSCLASSIFIER_TRAIN_KERNEL_H__

#include "numeric_table.h"
#include "model.h"
#include "algorithm.h"
#include "multi_class_classifier_train_types.h"
#include "service_defines.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace training
{
namespace internal
{
template <Method method, typename AlgorithmFPtype, typename ClsType, typename MccParType, CpuType cpu>
struct MultiClassClassifierTrainKernel : public Kernel
{
    services::Status compute(const NumericTable * a0, const NumericTable * a1, daal::algorithms::Model * r, const daal::algorithms::Parameter * par);
};

} // namespace internal

} // namespace training

} // namespace multi_class_classifier

} // namespace algorithms

} // namespace daal

#endif
