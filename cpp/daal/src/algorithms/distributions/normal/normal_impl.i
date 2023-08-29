/* file: normal_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of normal algorithm
//--
*/

#ifndef __NORMAL_IMPL_I__
#define __NORMAL_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace normal
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
Status NormalKernel<algorithmFPType, method, cpu>::compute(const normal::Parameter<algorithmFPType> * parameter, engines::BatchBase & engine,
                                                           NumericTable * resultTable)
{
    size_t nRows = resultTable->getNumberOfRows();

    daal::internal::WriteRows<algorithmFPType, cpu> resultBlock(resultTable, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(resultBlock);
    algorithmFPType * result = resultBlock.get();

    size_t size = nRows * resultTable->getNumberOfColumns();

    return compute(parameter, engine, size, result);
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status NormalKernel<algorithmFPType, method, cpu>::compute(const normal::Parameter<algorithmFPType> * parameter, engines::BatchBase & engine,
                                                           size_t n, algorithmFPType * resultArray)
{
    auto engineImpl = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl *>(&engine);
    DAAL_CHECK(engineImpl, ErrorIncorrectEngineParameter);

    return compute(parameter, *engineImpl, n, resultArray);
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status NormalKernel<algorithmFPType, method, cpu>::compute(const normal::Parameter<algorithmFPType> * parameter,
                                                           UniquePtr<engines::internal::BatchBaseImpl, cpu> & enginePtr, size_t n,
                                                           algorithmFPType * resultArray)
{
    return compute(parameter, *enginePtr, n, resultArray);
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status NormalKernel<algorithmFPType, method, cpu>::compute(const normal::Parameter<algorithmFPType> * parameter,
                                                           engines::internal::BatchBaseImpl & engine, size_t n, algorithmFPType * resultArray)
{
    algorithmFPType a     = parameter->a;
    algorithmFPType sigma = parameter->sigma;

    daal::internal::RNGsInst<algorithmFPType, cpu> rng;
    DAAL_CHECK(!rng.gaussian(n, resultArray, engine.getState(), a, sigma), ErrorIncorrectErrorcodeFromGenerator);
    return Status();
}

} // namespace internal
} // namespace normal
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
