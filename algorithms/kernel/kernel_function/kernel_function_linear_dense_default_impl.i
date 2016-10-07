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

template <typename algorithmFPType, CpuType cpu>
void KernelImplLinear<defaultDense, algorithmFPType, cpu>::prepareData(
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
void KernelImplLinear<defaultDense, algorithmFPType, cpu>::computeInternal(
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
void KernelImplLinear<defaultDense, algorithmFPType, cpu>::computeInternalVectorVector(
    size_t nFeatures, size_t nVectors1, const algorithmFPType *dataA1,
    size_t nVectors2, const algorithmFPType *dataA2,
    algorithmFPType *dataR, const ParameterBase *par)
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

template <typename algorithmFPType, CpuType cpu>
void KernelImplLinear<defaultDense, algorithmFPType, cpu>::computeInternalMatrixVector(
    size_t nFeatures, size_t nVectors1, const algorithmFPType *dataA1,
    size_t nVectors2, const algorithmFPType *dataA2,
    algorithmFPType *dataR, const ParameterBase *par)
{
    const Parameter *linPar = static_cast<const Parameter *>(par);
    algorithmFPType b = (algorithmFPType)(linPar->b);
    algorithmFPType k = (algorithmFPType)(linPar->k);
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

template <typename algorithmFPType, CpuType cpu>
void KernelImplLinear<defaultDense, algorithmFPType, cpu>::computeInternalMatrixMatrix(
    size_t nFeatures, size_t nVectors1, const algorithmFPType *dataA1,
    size_t nVectors2, const algorithmFPType *dataA2,
    algorithmFPType *dataR, const ParameterBase *par, bool inputTablesSame)
{
    char uplo, trans, notrans;
    algorithmFPType alpha, beta;
    const Parameter *linPar = static_cast<const Parameter *>(par);
    algorithmFPType b = (algorithmFPType)(linPar->b);
    algorithmFPType k = (algorithmFPType)(linPar->k);

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
            Blas<algorithmFPType, cpu>::xxgemm(&trans, &notrans, (DAAL_INT *)&nVectors2, (DAAL_INT *)&nVectors1, (DAAL_INT *)&nFeatures,
                                               &alpha, (algorithmFPType *)dataA2, (DAAL_INT *)&nFeatures, (algorithmFPType *)dataA1, (DAAL_INT *)&nFeatures, &beta,
                                               dataR, (DAAL_INT *)&nVectors2);
        }
        else
        {
            Blas<algorithmFPType, cpu>::xgemm(&trans, &notrans, (DAAL_INT *)&nVectors2, (DAAL_INT *)&nVectors1, (DAAL_INT *)&nFeatures,
                                              &alpha, (algorithmFPType *)dataA2, (DAAL_INT *)&nFeatures, (algorithmFPType *)dataA1, (DAAL_INT *)&nFeatures, &beta,
                                              dataR, (DAAL_INT *)&nVectors2);
        }
    }
    else
    {
        if (isInParallel)
        {
            Blas<algorithmFPType, cpu>::xxsyrk(&uplo, &trans, (DAAL_INT *)&nVectors1, (DAAL_INT *)&nFeatures,
                                               &alpha, (algorithmFPType *)dataA1, (DAAL_INT *)&nFeatures, &beta, dataR, (DAAL_INT *)&nVectors1);
        }
        else
        {
            Blas<algorithmFPType, cpu>::xsyrk(&uplo, &trans, (DAAL_INT *)&nVectors1, (DAAL_INT *)&nFeatures,
                                              &alpha, (algorithmFPType *)dataA1, (DAAL_INT *)&nFeatures, &beta, dataR, (DAAL_INT *)&nVectors1);
        }
        for (size_t i = 0; i < nVectors1; i++)
        {
            for (size_t j = i + 1; j < nVectors1; j++)
            {
                dataR[i * nVectors1 + j] = dataR[j * nVectors1 + i];
            }
        }
    }

    if (b != (algorithmFPType)0.0)
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
