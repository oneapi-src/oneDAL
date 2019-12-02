/* file: assoc_rules_kernel.h */
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
//  Declaration of template function that computes association rules results.
//--
*/

#ifndef __ASSOC_RULES_KERNEL_H__
#define __ASSOC_RULES_KERNEL_H__

#include "apriori_types.h"
#include "kernel.h"

#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace internal
{
/**
 *  Structure that contains kernels for association rules mining
 */
template <Method method, typename algorithmFPType, CpuType cpu>
class AssociationRulesKernel : public Kernel
{
public:
    /** Find "large" item sets and build association rules */
    services::Status compute(const NumericTable * a, NumericTable * r[], const daal::algorithms::Parameter * parameter);
};

} // namespace internal

} // namespace association_rules

} // namespace algorithms

} // namespace daal

#endif
