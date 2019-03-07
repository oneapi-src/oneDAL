/* file: uniform_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
Status UniformKernel<algorithmFPType, method, cpu>::compute(const uniform::Parameter<algorithmFPType> &parameter, engines::BatchBase &engine, NumericTable *resultTable)
{
    size_t nRows = resultTable->getNumberOfRows();

    daal::internal::WriteRows<algorithmFPType, cpu> resultBlock(resultTable, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(resultBlock);
    algorithmFPType *resultArray = resultBlock.get();

    size_t n = nRows * resultTable->getNumberOfColumns();

    return compute(parameter, engine, n, resultArray);
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status UniformKernel<algorithmFPType, method, cpu>::compute(const uniform::Parameter<algorithmFPType> &parameter, engines::BatchBase &engine, size_t n, algorithmFPType *resultArray)
{
    return compute(parameter.a, parameter.b, engine, n, resultArray);
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status UniformKernel<algorithmFPType, method, cpu>::compute(const uniform::Parameter<algorithmFPType> &parameter, UniquePtr<engines::internal::BatchBaseImpl, cpu> &enginePtr, size_t n, algorithmFPType *resultArray)
{
    return compute(parameter.a, parameter.b, *enginePtr, n, resultArray);
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status UniformKernel<algorithmFPType, method, cpu>::compute(algorithmFPType a, algorithmFPType b, engines::BatchBase &engine, size_t n, algorithmFPType *resultArray)
{
    auto engineImpl = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(&engine);
    return compute(a, b, *engineImpl, n, resultArray);
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status UniformKernel<algorithmFPType, method, cpu>::compute(algorithmFPType a, algorithmFPType b, engines::internal::BatchBaseImpl &engine, size_t n, algorithmFPType *resultArray)
{
    daal::internal::RNGs<algorithmFPType, cpu> rng;
    DAAL_CHECK(!rng.uniform(n, resultArray, engine.getState(), a, b), ErrorIncorrectErrorcodeFromGenerator);
    return Status();
}

} // namespace internal
} // namespace uniform
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
