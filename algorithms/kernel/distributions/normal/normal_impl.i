/* file: normal_impl.i */
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

template<typename algorithmFPType, Method method, CpuType cpu>
Status NormalKernel<algorithmFPType, method, cpu>::compute(const normal::Parameter<algorithmFPType> *parameter,
                                                           engines::BatchBase &engine,
                                                           NumericTable *resultTable)
{
    size_t nRows = resultTable->getNumberOfRows();

    daal::internal::WriteRows<algorithmFPType, cpu> resultBlock(resultTable, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(resultBlock);
    algorithmFPType *result = resultBlock.get();

    size_t size = nRows * resultTable->getNumberOfColumns();

    return compute(parameter, engine, size, result);
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status NormalKernel<algorithmFPType, method, cpu>::compute(const normal::Parameter<algorithmFPType> *parameter,
                                                           engines::BatchBase &engine, size_t n,
                                                           algorithmFPType *resultArray)
{
    auto engineImpl = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(&engine);
    DAAL_CHECK(engineImpl, ErrorIncorrectEngineParameter);

    return compute(parameter, *engineImpl, n, resultArray);
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status NormalKernel<algorithmFPType, method, cpu>::compute(const normal::Parameter<algorithmFPType> *parameter,
                                                           UniquePtr<engines::internal::BatchBaseImpl, cpu> &enginePtr,
                                                           size_t n, algorithmFPType *resultArray)
{
    return compute(parameter, *enginePtr, n, resultArray);
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status NormalKernel<algorithmFPType, method, cpu>::compute(const normal::Parameter<algorithmFPType> *parameter,
                                                           engines::internal::BatchBaseImpl &engine,
                                                           size_t n, algorithmFPType *resultArray)
{
    algorithmFPType a     = parameter->a;
    algorithmFPType sigma = parameter->sigma;

    daal::internal::RNGs<algorithmFPType, cpu> rng;
    DAAL_CHECK(!rng.gaussian(n, resultArray, engine.getState(), a, sigma), ErrorIncorrectErrorcodeFromGenerator);
    return Status();
}

} // namespace internal
} // namespace normal
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
