/* file: uniform_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of uniform algorithm
//--
*/

#ifndef __UNIFORM_IMPL_I__
#define __UNIFORM_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace uniform
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
Status UniformKernel<algorithmFPType, method, cpu>::compute(const uniform::Parameter<algorithmFPType> *parameter, NumericTable *resultTable)
{
    daal::algorithms::engines::internal::BatchBaseImpl* engine = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(parameter->engine.get());
    DAAL_CHECK(engine, ErrorIncorrectEngineParameter);

    size_t nRows = resultTable->getNumberOfRows();

    daal::internal::WriteRows<algorithmFPType, cpu> resultBlock(resultTable, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(resultBlock);
    algorithmFPType *result = resultBlock.get();

    size_t size = nRows * resultTable->getNumberOfColumns();

    algorithmFPType a = parameter->a;
    algorithmFPType b = parameter->b;

    daal::internal::RNGs<algorithmFPType, cpu> rng;
    DAAL_CHECK(!rng.uniform(size, result, engine->getState(), a, b), ErrorIncorrectErrorcodeFromGenerator);
    return Status();
}

} // namespace internal
} // namespace uniform
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
