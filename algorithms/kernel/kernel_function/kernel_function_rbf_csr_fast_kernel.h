/* file: kernel_function_rbf_csr_fast_kernel.h */
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
//  Declaration of template structs that calculate SVM RBF Kernel functions.
//--
*/

#ifndef __KERNEL_FUNCTION_RBF_CSR_FAST_KERNEL_H__
#define __KERNEL_FUNCTION_RBF_CSR_FAST_KERNEL_H__

#include "kernel_function_csr_base.h"
#include "kernel_function_rbf_base.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace rbf
{
namespace internal
{

template <typename AlgorithmFPType, CpuType cpu>
struct KernelImplRBF<fastCSR, AlgorithmFPType, cpu> :
        public daal::algorithms::kernel_function::internal::KernelCSRImplBase<AlgorithmFPType, cpu>
{
    using daal::algorithms::kernel_function::internal::KernelCSRImplBase<AlgorithmFPType, cpu>::prepareDataVectorVector;
    using daal::algorithms::kernel_function::internal::KernelCSRImplBase<AlgorithmFPType, cpu>::prepareDataMatrixVector;
    using daal::algorithms::kernel_function::internal::KernelCSRImplBase<AlgorithmFPType, cpu>::prepareDataMatrixMatrix;
    using daal::algorithms::kernel_function::internal::KernelCSRImplBase<AlgorithmFPType, cpu>::computeDotProduct;

    virtual void computeInternal(size_t nFeatures,
                                 size_t nVectors1, const AlgorithmFPType *dataA1, const size_t *colIndicesA1, const size_t *rowOffsetsA1,
                                 size_t nVectors2, const AlgorithmFPType *dataA2, const size_t *colIndicesA2, const size_t *rowOffsetsA2,
                                 AlgorithmFPType *dataR, const ParameterBase *par, bool inputTablesSame);

    virtual void prepareData(CSRBlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA1,
                             CSRBlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA2,
                             BlockMicroTable<AlgorithmFPType, writeOnly, cpu> &mtR,
                             const ParameterBase *svmPar,
                             size_t *nVectors1, AlgorithmFPType **dataA1, size_t **colIndicesA1, size_t **rowOffsetsA1,
                             size_t *nVectors2, AlgorithmFPType **dataA2, size_t **colIndicesA2, size_t **rowOffsetsA2,
                             AlgorithmFPType **dataR, bool inputTablesSame);

    void computeInternalVectorVector(size_t nFeatures,
            size_t nVectors1, const AlgorithmFPType *dataA1, const size_t *colIndicesA1, const size_t *rowOffsetsA1,
            size_t nVectors2, const AlgorithmFPType *dataA2, const size_t *colIndicesA2, const size_t *rowOffsetsA2,
            AlgorithmFPType *dataR, const ParameterBase *par);

    void computeInternalMatrixVector(size_t nFeatures,
            size_t nVectors1, const AlgorithmFPType *dataA1, const size_t *colIndicesA1, const size_t *rowOffsetsA1,
            size_t nVectors2, const AlgorithmFPType *dataA2, const size_t *colIndicesA2, const size_t *rowOffsetsA2,
            AlgorithmFPType *dataR, const ParameterBase *par);

    void computeInternalMatrixMatrix(size_t nFeatures,
            size_t nVectors1, const AlgorithmFPType *dataA1, const size_t *colIndicesA1, const size_t *rowOffsetsA1,
            size_t nVectors2, const AlgorithmFPType *dataA2, const size_t *colIndicesA2, const size_t *rowOffsetsA2,
            AlgorithmFPType *dataR, const ParameterBase *par, bool inputTablesSame);
};

} // namespace internal

} // namespace rbf

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif
