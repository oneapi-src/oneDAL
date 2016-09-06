/* file: assoc_rules_batch_container.h */
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
template<typename interm, Method method, CpuType cpu>
BatchContainer<interm, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv) : AnalysisContainerIface<batch>(daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::AssociationRulesKernel, method, interm);
}

template<typename interm, Method method, CpuType cpu>
BatchContainer<interm, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename interm, Method method, CpuType cpu>
void BatchContainer<interm, method, cpu>::compute()
{
    Result *result = static_cast<Result *>(_res);
    Input *input = static_cast<Input *>(_in);

    size_t nr = result->size();

    NumericTable *a0 = static_cast<NumericTable *>(input->get(data).get());

    Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(_par));

    NumericTable **r = new NumericTable*[nr];
    r[0] = static_cast<NumericTable *>(result->get(largeItemsets).get());
    r[1] = static_cast<NumericTable *>(result->get(largeItemsetsSupport).get());
    if(algParameter->discoverRules)
    {
        r[2] = static_cast<NumericTable *>(result->get(antecedentItemsets).get());
        r[3] = static_cast<NumericTable *>(result->get(consequentItemsets).get());
        r[4] = static_cast<NumericTable *>(result->get(confidence).get());
    }

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::AssociationRulesKernel, __DAAL_KERNEL_ARGUMENTS(method, interm), compute, a0, nr, r, algParameter);

    delete [] r;
}

} // namespace association_rules

} // namespace algorithms

} // namespace daal
