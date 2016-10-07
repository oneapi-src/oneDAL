/* file: sgd_types_fpt.cpp */
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
//  Implementation of sgd solver classes.
//--
*/

#include "algorithms/optimization_solver/sgd/sgd_types.h"
#include "data_management/data/memory_block.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace interface1
{

template <typename algorithmFPType>
void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    super::allocate<algorithmFPType>(input, par, method);
    const BaseParameter *algParam = static_cast<const BaseParameter *>(par);
    if(!algParam->optionalResultRequired)
        return;
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if(pOpt.get())
    {
        if(pOpt->size() != optionalDataSize)
            return;//error, will be found in check
    }
    else
    {
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(optionalDataSize));
        Argument::set(iterative_solver::optionalResult, pOpt);
    }
    data_management::MemoryBlockPtr pMem = data_management::MemoryBlock::cast(pOpt->get(rngState));
    if(!pMem.get())
    {
        if(!algParam->batchIndices.get())
        {
            pMem = data_management::MemoryBlockPtr(new data_management::MemoryBlock());
            pOpt->set(rngState, pMem);
        }
    }
}
template DAAL_EXPORT void Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

} // namespace interface1
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
