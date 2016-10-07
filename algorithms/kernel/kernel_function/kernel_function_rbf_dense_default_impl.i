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

template <typename algorithmFPType, CpuType cpu>
void KernelImplRBF<defaultDense, algorithmFPType, cpu>::prepareData(
    BlockMicroTable<algorithmFPType, readOnly,  cpu> &mtA1,
    BlockMicroTable<algorithmFPType, readOnly,  cpu> &mtA2,
    BlockMicroTable<algorithmFPType, writeOnly, cpu> &mtR,
    const ParameterBase *svmPar,
    size_t *nVectors1, algorithmFPType **dataA1,  size_t *nVectors2, algorithmFPType **dataA2,
    algorithmFPType **dataR, bool inputTablesSame)
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

template <typename algorithmFPType, CpuType cpu>
void KernelImplRBF<defaultDense, algorithmFPType, cpu>::computeInternal(
    size_t nFeatures, size_t nVectors1,
    const algorithmFPType *dataA1, size_t nVectors2, const algorithmFPType *dataA2,
    algorithmFPType *dataR, const ParameterBase *par, bool inputTablesSame)
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

template <typename algorithmFPType, CpuType cpu>
void KernelImplRBF<defaultDense, algorithmFPType, cpu>::computeInternalVectorVector(
    size_t nFeatures, size_t nVectors1, const algorithmFPType *dataA1,
    size_t nVectors2, const algorithmFPType *dataA2,
    algorithmFPType *dataR, const ParameterBase *par)
{
    const Parameter *rbfPar = static_cast<const Parameter *>(par);
    algorithmFPType invSqrSigma = (algorithmFPType)(1.0 / (rbfPar->sigma * rbfPar->sigma));
    algorithmFPType factor = 0.0;
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nFeatures; i++)
    {
        algorithmFPType diff = (dataA1[i] - dataA2[i]);
        factor += diff * diff;
    }
    factor *= -0.5 * invSqrSigma;
    daal::internal::Math<algorithmFPType, cpu>::vExp(1, &factor, dataR);
}

template <typename algorithmFPType, CpuType cpu>
void KernelImplRBF<defaultDense, algorithmFPType, cpu>::computeInternalMatrixVector(
    size_t nFeatures, size_t nVectors1, const algorithmFPType *dataA1,
    size_t nVectors2, const algorithmFPType *dataA2,
    algorithmFPType *dataR, const ParameterBase *par)
{
    const Parameter *rbfPar = static_cast<const Parameter *>(par);
    algorithmFPType invSqrSigma = (algorithmFPType)(1.0 / (rbfPar->sigma * rbfPar->sigma));
    for (size_t i = 0; i < nVectors1; i++)
    {
        algorithmFPType factor = 0.0;
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nFeatures; j++)
        {
            algorithmFPType diff = (dataA1[i * nFeatures + j] - dataA2[j]);
            factor += diff * diff;
        }
        dataR[i] = -0.5 * invSqrSigma * factor;
    }
    daal::internal::Math<algorithmFPType, cpu>::vExp(nVectors1, dataR, dataR);
}

template <typename algorithmFPType, CpuType cpu>
void KernelImplRBF<defaultDense, algorithmFPType, cpu>::computeInternalMatrixMatrix(
    size_t nFeatures, size_t nVectors1, const algorithmFPType *dataA1,
    size_t nVectors2, const algorithmFPType *dataA2,
    algorithmFPType *dataR, const ParameterBase *par, bool inputTablesSame)
{
    const Parameter *rbfPar = static_cast<const Parameter *>(par);
    algorithmFPType coeff = (algorithmFPType)(-0.5 / (rbfPar->sigma * rbfPar->sigma));

    bool isInParallel = is_in_parallel();
    if (!inputTablesSame)
    {
        char trans, notrans;
        algorithmFPType zero = 0.0, negTwo = -2.0;
        trans = 'T';
        notrans = 'N';
        if (isInParallel)
        {
            Blas<algorithmFPType, cpu>::xxgemm(&trans, &notrans, (DAAL_INT *)&nVectors2, (DAAL_INT *)&nVectors1, (DAAL_INT *)&nFeatures,
                                               &negTwo, (algorithmFPType *)dataA2, (DAAL_INT *)&nFeatures, (algorithmFPType *)dataA1, (DAAL_INT *)&nFeatures, &zero,
                                               dataR, (DAAL_INT *)&nVectors2);
        }
        else
        {
            Blas<algorithmFPType, cpu>::xgemm(&trans, &notrans, (DAAL_INT *)&nVectors2, (DAAL_INT *)&nVectors1, (DAAL_INT *)&nFeatures,
                                              &negTwo, (algorithmFPType *)dataA2, (DAAL_INT *)&nFeatures, (algorithmFPType *)dataA1, (DAAL_INT *)&nFeatures, &zero,
                                              dataR, (DAAL_INT *)&nVectors2);
        }
        algorithmFPType *buffer = (algorithmFPType *)daal::services::daal_malloc((nVectors1 + nVectors2) * sizeof(algorithmFPType));
        if (!buffer) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }
        algorithmFPType *sqrDataA1 = buffer;
        algorithmFPType *sqrDataA2 = buffer + nVectors1;
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
        daal::internal::Math<algorithmFPType, cpu>::vExp(nVectors1 * nVectors2, dataR, dataR);
        daal::services::daal_free(buffer);
    }
    else
    {
        char uplo, trans;
        algorithmFPType zero = 0.0, one = 1.0, two = 2.0;
        uplo = 'U';
        trans = 'T';
        if (isInParallel)
        {
            Blas<algorithmFPType, cpu>::xxsyrk(&uplo, &trans, (DAAL_INT *)&nVectors1, (DAAL_INT *)&nFeatures,
                                               &one, (algorithmFPType *)dataA1, (DAAL_INT *)&nFeatures, &zero, dataR, (DAAL_INT *)&nVectors1);
        }
        else
        {
            Blas<algorithmFPType, cpu>::xsyrk(&uplo, &trans, (DAAL_INT *)&nVectors1, (DAAL_INT *)&nFeatures,
                                              &one, (algorithmFPType *)dataA1, (DAAL_INT *)&nFeatures, &zero, dataR, (DAAL_INT *)&nVectors1);
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
            daal::internal::Math<algorithmFPType, cpu>::vExp(i + 1, dataR + i * nVectors1, dataR + i * nVectors1);
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
