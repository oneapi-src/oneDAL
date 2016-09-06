/* file: kernel_function_rbf_dense_default_impl.i */
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
//  RBF kernel functions implementation
//--
*/

#ifndef __KERNEL_FUNCTION_RBF_DENSE_DEFAULT_IMPL_I__
#define __KERNEL_FUNCTION_RBF_DENSE_DEFAULT_IMPL_I__

#include "kernel_function_types_rbf.h"

#include "service_micro_table.h"
#include "service_math.h"
#include "service_blas.h"
#include "threading.h"

using namespace daal::data_management;

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
void KernelImplRBF<defaultDense, AlgorithmFPType, cpu>::prepareData(
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
void KernelImplRBF<defaultDense, AlgorithmFPType, cpu>::computeInternal(
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
void KernelImplRBF<defaultDense, AlgorithmFPType, cpu>::computeInternalVectorVector(
            size_t nFeatures, size_t nVectors1, const AlgorithmFPType *dataA1,
            size_t nVectors2, const AlgorithmFPType *dataA2,
            AlgorithmFPType *dataR, const ParameterBase *par)
{
    const Parameter *rbfPar = static_cast<const Parameter *>(par);
    AlgorithmFPType invSqrSigma = (AlgorithmFPType)(1.0 / (rbfPar->sigma * rbfPar->sigma));
    AlgorithmFPType factor = 0.0;
  PRAGMA_IVDEP
  PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nFeatures; i++)
    {
        AlgorithmFPType diff = (dataA1[i] - dataA2[i]);
        factor += diff * diff;
    }
    factor *= -0.5 * invSqrSigma;
    daal::internal::Math<AlgorithmFPType,cpu>::vExp(1, &factor, dataR);
}

template <typename AlgorithmFPType, CpuType cpu>
void KernelImplRBF<defaultDense, AlgorithmFPType, cpu>::computeInternalMatrixVector(
            size_t nFeatures, size_t nVectors1, const AlgorithmFPType *dataA1,
            size_t nVectors2, const AlgorithmFPType *dataA2,
            AlgorithmFPType *dataR, const ParameterBase *par)
{
    const Parameter *rbfPar = static_cast<const Parameter *>(par);
    AlgorithmFPType invSqrSigma = (AlgorithmFPType)(1.0 / (rbfPar->sigma * rbfPar->sigma));
    for (size_t i = 0; i < nVectors1; i++)
    {
        AlgorithmFPType factor = 0.0;
      PRAGMA_IVDEP
      PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nFeatures; j++)
        {
            AlgorithmFPType diff = (dataA1[i * nFeatures + j] - dataA2[j]);
            factor += diff * diff;
        }
        dataR[i] = -0.5 * invSqrSigma * factor;
    }
    daal::internal::Math<AlgorithmFPType,cpu>::vExp(nVectors1, dataR, dataR);
}

template <typename AlgorithmFPType, CpuType cpu>
void KernelImplRBF<defaultDense, AlgorithmFPType, cpu>::computeInternalMatrixMatrix(
            size_t nFeatures, size_t nVectors1, const AlgorithmFPType *dataA1,
            size_t nVectors2, const AlgorithmFPType *dataA2,
            AlgorithmFPType *dataR, const ParameterBase *par, bool inputTablesSame)
{
    const Parameter *rbfPar = static_cast<const Parameter *>(par);
    AlgorithmFPType coeff = (AlgorithmFPType)(-0.5 / (rbfPar->sigma * rbfPar->sigma));

    bool isInParallel = is_in_parallel();
    if (!inputTablesSame)
    {
        char trans, notrans;
        AlgorithmFPType zero = 0.0, negTwo = -2.0;
        trans = 'T';
        notrans = 'N';
        if (isInParallel)
        {
            Blas<AlgorithmFPType, cpu>::xxgemm(&trans, &notrans, (MKL_INT *)&nVectors2, (MKL_INT *)&nVectors1, (MKL_INT *)&nFeatures,
                &negTwo, (AlgorithmFPType *)dataA2, (MKL_INT *)&nFeatures, (AlgorithmFPType *)dataA1, (MKL_INT *)&nFeatures, &zero,
                dataR, (MKL_INT *)&nVectors2);
        }
        else
        {
            Blas<AlgorithmFPType, cpu>::xgemm(&trans, &notrans, (MKL_INT *)&nVectors2, (MKL_INT *)&nVectors1, (MKL_INT *)&nFeatures,
                &negTwo, (AlgorithmFPType *)dataA2, (MKL_INT *)&nFeatures, (AlgorithmFPType *)dataA1, (MKL_INT *)&nFeatures, &zero,
                dataR, (MKL_INT *)&nVectors2);
        }
        AlgorithmFPType *buffer = (AlgorithmFPType *)daal::services::daal_malloc((nVectors1 + nVectors2) * sizeof(AlgorithmFPType));
        if (!buffer) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }
        AlgorithmFPType *sqrDataA1 = buffer;
        AlgorithmFPType *sqrDataA2 = buffer + nVectors1;
        for (size_t i = 0; i < nVectors1; i++)
        {
            sqrDataA1[i] = zero;
            for (size_t j = 0; j < nFeatures; j++)
            {
                sqrDataA1[i] += dataA1[i * nFeatures + j] * dataA1[i * nFeatures + j];
            }
        }
        for (size_t i = 0; i < nVectors2; i++)
        {
            sqrDataA2[i] = zero;
            for (size_t j = 0; j < nFeatures; j++)
            {
                sqrDataA2[i] += dataA2[i * nFeatures + j] * dataA2[i * nFeatures + j];
            }
        }
        for (size_t i = 0; i < nVectors1; i++)
        {
            for (size_t k = 0; k < nVectors2; k++)
            {
                dataR[i * nVectors2 + k] += (sqrDataA1[i] + sqrDataA2[k]);
                dataR[i * nVectors2 + k] *= coeff;
            }
        }
        daal::internal::Math<AlgorithmFPType,cpu>::vExp(nVectors1 * nVectors2, dataR, dataR);
        daal::services::daal_free(buffer);
    }
    else
    {
        char uplo, trans;
        AlgorithmFPType zero = 0.0, one = 1.0, two = 2.0;
        uplo = 'U';
        trans = 'T';
        if (isInParallel)
        {
            Blas<AlgorithmFPType, cpu>::xxsyrk(&uplo, &trans, (MKL_INT *)&nVectors1, (MKL_INT *)&nFeatures,
                &one, (AlgorithmFPType *)dataA1, (MKL_INT *)&nFeatures, &zero, dataR, (MKL_INT *)&nVectors1);
        }
        else
        {
            Blas<AlgorithmFPType, cpu>::xsyrk(&uplo, &trans, (MKL_INT *)&nVectors1, (MKL_INT *)&nFeatures,
                &one, (AlgorithmFPType *)dataA1, (MKL_INT *)&nFeatures, &zero, dataR, (MKL_INT *)&nVectors1);
        }
        for (size_t i = 0; i < nVectors1; i++)
        {
            for (size_t k = 0; k < i; k++)
            {
                dataR[i * nVectors1 + k] = coeff * (dataR[i * nVectors1 + i] + dataR[k * nVectors1 + k] -
                        two * dataR[i * nVectors1 + k]);
            }
        }
        for (size_t i = 0; i < nVectors1; i++)
        {
            dataR[i * nVectors1 + i] = zero;
            daal::internal::Math<AlgorithmFPType,cpu>::vExp(i + 1, dataR + i * nVectors1, dataR + i * nVectors1);
        }
        for (size_t i = 0; i < nVectors1; i++)
        {
            for (size_t k = i + 1; k < nVectors1; k++)
            {
                dataR[i * nVectors1 + k] = dataR[k * nVectors1 + i];
            }
        }
    }
}

} // namespace internal

} // namespace rbf

} // namespace kernel_function

} // namespace algorithms

} // namespace daal


#endif
