/* file: assoc_rules_batch_container.h */
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
//  Implementation of association rules mining algorithm container.
//--
*/

#include "apriori.h"
#include "assoc_rules_kernel.h"
#include "assoc_rules_apriori_kernel.h"

namespace daal
{
namespace algorithms
{
namespace association_rules
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : AnalysisContainerIface<batch>(daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::AssociationRulesKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Result * result = static_cast<Result *>(_res);
    Input * input   = static_cast<Input *>(_in);

    NumericTable * a0 = input->get(data).get();

    Parameter * algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(_par));

    NumericTable * r[lastResultId + 1];
    r[largeItemsets]        = result->get(largeItemsets).get();
    r[largeItemsetsSupport] = result->get(largeItemsetsSupport).get();
    r[antecedentItemsets]   = algParameter->discoverRules ? result->get(antecedentItemsets).get() : 0;
    r[consequentItemsets]   = algParameter->discoverRules ? result->get(consequentItemsets).get() : 0;
    r[confidence]           = algParameter->discoverRules ? result->get(confidence).get() : 0;

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::AssociationRulesKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a0, r, algParameter);
}

} // namespace association_rules

} // namespace algorithms

} // namespace daal
