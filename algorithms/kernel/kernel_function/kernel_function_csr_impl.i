/* file: kernel_function_csr_impl.i */
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
//  Common kernel functions implementation
//--
*/

#ifndef __KERNEL_FUNCTION_CSR_IMPL_I__
#define __KERNEL_FUNCTION_CSR_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace internal
{

template <typename AlgorithmFPType, CpuType cpu>
AlgorithmFPType KernelCSRImplBase<AlgorithmFPType, cpu>::computeDotProduct(
            size_t startIndex1, size_t endIndex1, const AlgorithmFPType *dataA1, const size_t *colIndicesA1,
            size_t startIndex2, size_t endIndex2, const AlgorithmFPType *dataA2, const size_t *colIndicesA2)
{
    size_t indexA1 = startIndex1;
    size_t indexA2 = startIndex2;
    AlgorithmFPType sum = 0.0;
    while ((indexA1 < endIndex1) && (indexA2 < endIndex2))
    {
        size_t colIndex1 = colIndicesA1[indexA1];
        size_t colIndex2 = colIndicesA2[indexA2];
        if (colIndex1 == colIndex2)
        {
            sum += dataA1[indexA1] * dataA2[indexA2];
            indexA1++;
            indexA2++;
        }
        else if (colIndex1 > colIndex2)
        {
            indexA2++;
        }
        else // (colIndex1 < colIndex2)
        {
            indexA1++;
        }
    }
    return sum;
}

}
}
}
}

#endif
