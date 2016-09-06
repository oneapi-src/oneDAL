/* file: kernel_function_linear_dense_default_kernel.h */
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

#ifndef __KERNEL_FUNCTION_DENSE_KERNEL_H__
#define __KERNEL_FUNCTION_DENSE_KERNEL_H__

#include "kernel_function_dense_base.h"
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

template <typename AlgorithmFPType, CpuType cpu>
struct KernelImplLinear<defaultDense, AlgorithmFPType, cpu> :
        public daal::algorithms::kernel_function::internal::KernelImplBase<AlgorithmFPType, cpu>
{
    using daal::algorithms::kernel_function::internal::KernelImplBase<AlgorithmFPType, cpu>::prepareDataVectorVector;
    using daal::algorithms::kernel_function::internal::KernelImplBase<AlgorithmFPType, cpu>::prepareDataMatrixVector;
    using daal::algorithms::kernel_function::internal::KernelImplBase<AlgorithmFPType, cpu>::prepareDataMatrixMatrix;

    virtual void computeInternal(size_t nFeatures, size_t nVectors1, const AlgorithmFPType *dataA1,
                                 size_t nVectors2, const AlgorithmFPType *dataA2,
                                 AlgorithmFPType *dataR, const ParameterBase *par, bool inputTablesSame);

    virtual void prepareData(BlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA1,
                             BlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA2,
                             BlockMicroTable<AlgorithmFPType, writeOnly, cpu> &mtR,
                             const ParameterBase *svmPar,
                             size_t *nVectors1, AlgorithmFPType **dataA1,  size_t *nVectors2, AlgorithmFPType **dataA2,
                             AlgorithmFPType **dataR, bool inputTablesSame);

    void computeInternalVectorVector(
            size_t nFeatures, size_t nVectors1, const AlgorithmFPType *dataA1,
            size_t nVectors2, const AlgorithmFPType *dataA2,
            AlgorithmFPType *dataR, const ParameterBase *par);

    void computeInternalMatrixVector(
            size_t nFeatures, size_t nVectors1, const AlgorithmFPType *dataA1,
            size_t nVectors2, const AlgorithmFPType *dataA2,
            AlgorithmFPType *dataR, const ParameterBase *par);

    void computeInternalMatrixMatrix(
            size_t nFeatures, size_t nVectors1, const AlgorithmFPType *dataA1,
            size_t nVectors2, const AlgorithmFPType *dataA2,
            AlgorithmFPType *dataR, const ParameterBase *par, bool inputTablesSame);
};

} //internal

} //linear

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif
