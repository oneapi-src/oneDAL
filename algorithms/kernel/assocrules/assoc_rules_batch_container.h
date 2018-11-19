/* file: assoc_rules_batch_container.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv) : AnalysisContainerIface<batch>(daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::AssociationRulesKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Result *result = static_cast<Result *>(_res);
    Input *input = static_cast<Input *>(_in);

    NumericTable *a0 = input->get(data).get();

    Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(_par));

    NumericTable* r[lastResultId + 1];
    r[largeItemsets] = result->get(largeItemsets).get();
    r[largeItemsetsSupport] = result->get(largeItemsetsSupport).get();
    r[antecedentItemsets] = algParameter->discoverRules ? result->get(antecedentItemsets).get() : 0;
    r[consequentItemsets] = algParameter->discoverRules ? result->get(consequentItemsets).get() : 0;
    r[confidence] = algParameter->discoverRules ? result->get(confidence).get() : 0;

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::AssociationRulesKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a0, r, algParameter);
}

} // namespace association_rules

} // namespace algorithms

} // namespace daal
