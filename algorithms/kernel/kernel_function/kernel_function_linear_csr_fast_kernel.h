/* file: kernel_function_linear_csr_fast_kernel.h */
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
//  Declaration of template structs that calculate SVM Linear Kernel functions.
//--
*/

#ifndef __KERNEL_FUNCTION_LINEAR_CSR_KERNEL_H__
#define __KERNEL_FUNCTION_LINEAR_CSR_KERNEL_H__

#include "kernel_function_csr_base.h"
#include "kernel_function_linear_base.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace linear
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
struct KernelImplLinear<fastCSR, algorithmFPType, cpu> :
    public daal::algorithms::kernel_function::internal::KernelCSRImplBase<algorithmFPType, cpu>
{
    using daal::algorithms::kernel_function::internal::KernelCSRImplBase<algorithmFPType, cpu>::prepareDataVectorVector;
    using daal::algorithms::kernel_function::internal::KernelCSRImplBase<algorithmFPType, cpu>::prepareDataMatrixVector;
    using daal::algorithms::kernel_function::internal::KernelCSRImplBase<algorithmFPType, cpu>::prepareDataMatrixMatrix;
    using daal::algorithms::kernel_function::internal::KernelCSRImplBase<algorithmFPType, cpu>::computeDotProduct;

    virtual void computeInternal(size_t nFeatures,
                                 size_t nVectors1, const algorithmFPType *dataA1, const size_t *colIndicesA1, const size_t *rowOffsetsA1,
                                 size_t nVectors2, const algorithmFPType *dataA2, const size_t *colIndicesA2, const size_t *rowOffsetsA2,
                                 algorithmFPType *dataR, const ParameterBase *par, bool inputTablesSame);

    virtual void prepareData(CSRBlockMicroTable<algorithmFPType, readOnly,  cpu> &mtA1,
                             CSRBlockMicroTable<algorithmFPType, readOnly,  cpu> &mtA2,
                             BlockMicroTable<algorithmFPType, writeOnly, cpu> &mtR,
                             const ParameterBase *svmPar,
                             size_t *nVectors1, algorithmFPType **dataA1, size_t **colIndicesA1, size_t **rowOffsetsA1,
                             size_t *nVectors2, algorithmFPType **dataA2, size_t **colIndicesA2, size_t **rowOffsetsA2,
                             algorithmFPType **dataR, bool inputTablesSame);

    void computeInternalVectorVector(size_t nFeatures,
                                     size_t nVectors1, const algorithmFPType *dataA1, const size_t *colIndicesA1, const size_t *rowOffsetsA1,
                                     size_t nVectors2, const algorithmFPType *dataA2, const size_t *colIndicesA2, const size_t *rowOffsetsA2,
                                     algorithmFPType *dataR, const ParameterBase *par);

    void computeInternalMatrixVector(size_t nFeatures,
                                     size_t nVectors1, const algorithmFPType *dataA1, const size_t *colIndicesA1, const size_t *rowOffsetsA1,
                                     size_t nVectors2, const algorithmFPType *dataA2, const size_t *colIndicesA2, const size_t *rowOffsetsA2,
                                     algorithmFPType *dataR, const ParameterBase *par);

    void computeInternalMatrixMatrix(size_t nFeatures,
                                     size_t nVectors1, const algorithmFPType *dataA1, const size_t *colIndicesA1, const size_t *rowOffsetsA1,
                                     size_t nVectors2, const algorithmFPType *dataA2, const size_t *colIndicesA2, const size_t *rowOffsetsA2,
                                     algorithmFPType *dataR, const ParameterBase *par, bool inputTablesSame);
};

} // namespace internal

} // namespace linear

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif
