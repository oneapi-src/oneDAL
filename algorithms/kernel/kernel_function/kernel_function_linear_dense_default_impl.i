/* file: kernel_function_linear_dense_default_impl.i */
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
//  Linear kernel functions implementation
//--
*/

#ifndef __KERNEL_FUNCTION_LINEAR_DENSE_DEFAULT_IMPL_I__
#define __KERNEL_FUNCTION_LINEAR_DENSE_DEFAULT_IMPL_I__

#include "kernel_function_types_linear.h"

#include "service_blas.h"
#include "service_stat.h"
#include "service_micro_table.h"
#include "threading.h"

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
void KernelImplLinear<defaultDense, AlgorithmFPType, cpu>::prepareData(
            BlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA1,
            BlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA2,
            BlockMicroTable<AlgorithmFPType, writeOnly, cpu> &mtR,
            const ParameterBase *svmPar,
            size_t *nVectors1, AlgorithmFPType **dataA1,  size_t *nVectors2, AlgorithmFPType **dataA2,
            AlgorithmFPType **dataR, bool inputTablesSame)
{
    if(this->_computationMode == vectorVector)
    {
        prepareDataVectorVector(mtA1, mtA2, mtR, svmPar, nVectors1, dataA1, nVectors2, dataA2, dataR, inputTablesSame);
    }
    else if(this->_computationMode == matrixVector)
    {
        prepareDataMatrixVector(mtA1, mtA2, mtR, svmPar, nVectors1, dataA1, nVectors2, dataA2, dataR, inputTablesSame);
    }
    else if(this->_computationMode == matrixMatrix)
    {
        prepareDataMatrixMatrix(mtA1, mtA2, mtR, svmPar, nVectors1, dataA1, nVectors2, dataA2, dataR, inputTablesSame);
    }
}

template <typename AlgorithmFPType, CpuType cpu>
void KernelImplLinear<defaultDense, AlgorithmFPType, cpu>::computeInternal(
            size_t nFeatures, size_t nVectors1,
            const AlgorithmFPType *dataA1, size_t nVectors2, const AlgorithmFPType *dataA2,
            AlgorithmFPType *dataR, const ParameterBase *par, bool inputTablesSame)
{
    if(this->_computationMode == vectorVector)
    {
        computeInternalVectorVector(nFeatures, nVectors1, dataA1, nVectors2, dataA2, dataR, par);
    }
    else if(this->_computationMode == matrixVector)
    {
        computeInternalMatrixVector(nFeatures, nVectors1, dataA1, nVectors2, dataA2, dataR, par);
    }
    else if(this->_computationMode == matrixMatrix)
    {
        computeInternalMatrixMatrix(nFeatures, nVectors1, dataA1, nVectors2, dataA2, dataR, par, inputTablesSame);
    }
}

template <typename AlgorithmFPType, CpuType cpu>
void KernelImplLinear<defaultDense, AlgorithmFPType, cpu>::computeInternalVectorVector(
            size_t nFeatures, size_t nVectors1, const AlgorithmFPType *dataA1,
            size_t nVectors2, const AlgorithmFPType *dataA2,
            AlgorithmFPType *dataR, const ParameterBase *par)
{
    const Parameter *linPar = static_cast<const Parameter *>(par);
    dataR[0] = 0.0;
  PRAGMA_IVDEP
  PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nFeatures; i++)
    {
        dataR[0] += dataA1[i] * dataA2[i];
    }
    dataR[0] = dataR[0] * linPar->k + linPar->b;
}

template <typename AlgorithmFPType, CpuType cpu>
void KernelImplLinear<defaultDense, AlgorithmFPType, cpu>::computeInternalMatrixVector(
            size_t nFeatures, size_t nVectors1, const AlgorithmFPType *dataA1,
            size_t nVectors2, const AlgorithmFPType *dataA2,
            AlgorithmFPType *dataR, const ParameterBase *par)
{
    const Parameter *linPar = static_cast<const Parameter *>(par);
    AlgorithmFPType b = (AlgorithmFPType)(linPar->b);
    AlgorithmFPType k = (AlgorithmFPType)(linPar->k);
    for (size_t i = 0; i < nVectors1; i++)
    {
        dataR[i] = 0.0;
  PRAGMA_IVDEP
  PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nFeatures; j++)
        {
            dataR[i] += dataA1[i * nFeatures + j] * dataA2[j];
        }
        dataR[i] = k * dataR[i];
        dataR[i] += b;
    }
}

template <typename AlgorithmFPType, CpuType cpu>
void KernelImplLinear<defaultDense, AlgorithmFPType, cpu>::computeInternalMatrixMatrix(
            size_t nFeatures, size_t nVectors1, const AlgorithmFPType *dataA1,
            size_t nVectors2, const AlgorithmFPType *dataA2,
            AlgorithmFPType *dataR, const ParameterBase *par, bool inputTablesSame)
{
    char uplo, trans, notrans;
    AlgorithmFPType alpha, beta;
    const Parameter *linPar = static_cast<const Parameter *>(par);
    AlgorithmFPType b = (AlgorithmFPType)(linPar->b);
    AlgorithmFPType k = (AlgorithmFPType)(linPar->k);

    /* Calculate X*Y' */
    uplo  = 'U';
    trans = 'T';
    notrans = 'N';
    alpha = k;
    beta  = 0.0;

    bool isInParallel = is_in_parallel();
    if (!inputTablesSame)
    {
        if (isInParallel)
        {
            Blas<AlgorithmFPType, cpu>::xxgemm(&trans, &notrans, (MKL_INT *)&nVectors2, (MKL_INT *)&nVectors1, (MKL_INT *)&nFeatures,
                &alpha, (AlgorithmFPType *)dataA2, (MKL_INT *)&nFeatures, (AlgorithmFPType *)dataA1, (MKL_INT *)&nFeatures, &beta,
                dataR, (MKL_INT *)&nVectors2);
        }
        else
        {
            Blas<AlgorithmFPType, cpu>::xgemm(&trans, &notrans, (MKL_INT *)&nVectors2, (MKL_INT *)&nVectors1, (MKL_INT *)&nFeatures,
                &alpha, (AlgorithmFPType *)dataA2, (MKL_INT *)&nFeatures, (AlgorithmFPType *)dataA1, (MKL_INT *)&nFeatures, &beta,
                dataR, (MKL_INT *)&nVectors2);
        }
    }
    else
    {
        if (isInParallel)
        {
            Blas<AlgorithmFPType, cpu>::xxsyrk(&uplo, &trans, (MKL_INT *)&nVectors1, (MKL_INT *)&nFeatures,
                &alpha, (AlgorithmFPType *)dataA1, (MKL_INT *)&nFeatures, &beta, dataR, (MKL_INT *)&nVectors1);
        }
        else
        {
            Blas<AlgorithmFPType, cpu>::xsyrk(&uplo, &trans, (MKL_INT *)&nVectors1, (MKL_INT *)&nFeatures,
                &alpha, (AlgorithmFPType *)dataA1, (MKL_INT *)&nFeatures, &beta, dataR, (MKL_INT *)&nVectors1);
        }
        for (size_t i = 0; i < nVectors1; i++)
        {
            for (size_t j = i + 1; j < nVectors1; j++)
            {
                dataR[i * nVectors1 + j] = dataR[j * nVectors1 + i];
            }
        }
    }

    if (b != (AlgorithmFPType)0.0)
    {
        size_t length = nVectors1 * nVectors2;
  PRAGMA_IVDEP
  PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < length; i++)
        {
            dataR[i] = dataR[i] + b;
        }
    }
}

} // namespace internal

} // namespace linear

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif
