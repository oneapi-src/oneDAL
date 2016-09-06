/* file: kernel_function_rbf_csr_fast_impl.i */
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

#ifndef __KERNEL_FUNCTION_RBF_CSR_FAST_IMPL_I__
#define __KERNEL_FUNCTION_RBF_CSR_FAST_IMPL_I__

#include "kernel_function_types_rbf.h"

#include "service_micro_table.h"
#include "service_math.h"

#include "kernel_function_csr_impl.i"

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
void KernelImplRBF<fastCSR, AlgorithmFPType, cpu>::prepareData(
            CSRBlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA1,
            CSRBlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA2,
            BlockMicroTable<AlgorithmFPType, writeOnly, cpu> &mtR,
            const ParameterBase *svmPar,
            size_t *nVectors1, AlgorithmFPType **dataA1, size_t **colIndicesA1, size_t **rowOffsetsA1,
            size_t *nVectors2, AlgorithmFPType **dataA2, size_t **colIndicesA2, size_t **rowOffsetsA2,
            AlgorithmFPType **dataR, bool inputTablesSame)
{
    if(this->_computationMode == vectorVector)
    {
        prepareDataVectorVector(mtA1, mtA2, mtR, svmPar, nVectors1, dataA1, colIndicesA1, rowOffsetsA1,
                                nVectors2, dataA2, colIndicesA2, rowOffsetsA2, dataR, inputTablesSame);
    }
    else if(this->_computationMode == matrixVector)
    {
        prepareDataMatrixVector(mtA1, mtA2, mtR, svmPar, nVectors1, dataA1, colIndicesA1, rowOffsetsA1,
                                nVectors2, dataA2, colIndicesA2, rowOffsetsA2, dataR, inputTablesSame);
    }
    else if(this->_computationMode == matrixMatrix)
    {
        prepareDataMatrixMatrix(mtA1, mtA2, mtR, svmPar, nVectors1, dataA1, colIndicesA1, rowOffsetsA1,
                                nVectors2, dataA2, colIndicesA2, rowOffsetsA2, dataR, inputTablesSame);
    }
}

template <typename AlgorithmFPType, CpuType cpu>
void KernelImplRBF<fastCSR, AlgorithmFPType, cpu>::computeInternal(
            size_t nFeatures,
            size_t nVectors1, const AlgorithmFPType *dataA1, const size_t *colIndicesA1, const size_t *rowOffsetsA1,
            size_t nVectors2, const AlgorithmFPType *dataA2, const size_t *colIndicesA2, const size_t *rowOffsetsA2,
            AlgorithmFPType *dataR, const ParameterBase *par, bool inputTablesSame)
{
    if(this->_computationMode == vectorVector)
    {
        computeInternalVectorVector(nFeatures, nVectors1, dataA1, colIndicesA1, rowOffsetsA1,
                                    nVectors2, dataA2, colIndicesA2, rowOffsetsA2, dataR, par);
    }
    else if(this->_computationMode == matrixVector)
    {
        computeInternalMatrixVector(nFeatures, nVectors1, dataA1, colIndicesA1, rowOffsetsA1,
                                    nVectors2, dataA2, colIndicesA2, rowOffsetsA2, dataR, par);
    }
    else if(this->_computationMode == matrixMatrix)
    {
        computeInternalMatrixMatrix(nFeatures, nVectors1, dataA1, colIndicesA1, rowOffsetsA1,
                                    nVectors2, dataA2, colIndicesA2, rowOffsetsA2, dataR, par, inputTablesSame);
    }
}

template <typename AlgorithmFPType, CpuType cpu>
void KernelImplRBF<fastCSR, AlgorithmFPType, cpu>::computeInternalVectorVector(
            size_t nFeatures,
            size_t nVectors1, const AlgorithmFPType *dataA1, const size_t *colIndicesA1, const size_t *rowOffsetsA1,
            size_t nVectors2, const AlgorithmFPType *dataA2, const size_t *colIndicesA2, const size_t *rowOffsetsA2,
            AlgorithmFPType *dataR, const ParameterBase *par)
{
    const Parameter *rbfPar = static_cast<const Parameter *>(par);
    AlgorithmFPType coeff = (AlgorithmFPType)(-0.5 / (rbfPar->sigma * rbfPar->sigma));
    size_t startIndex1 = rowOffsetsA1[0] - 1;
    size_t startIndex2 = rowOffsetsA2[0] - 1;
    size_t endIndex1   = rowOffsetsA1[1] - 1;
    size_t endIndex2   = rowOffsetsA2[1] - 1;
    AlgorithmFPType factor = computeDotProduct(startIndex1, endIndex1, dataA1, colIndicesA1,
                                               startIndex2, endIndex2, dataA2, colIndicesA2);
    factor *= -2.0;

    for (size_t index = startIndex1; index < endIndex1; index++)
    {
        factor += dataA1[index] * dataA1[index];
    }
    for (size_t index = startIndex2; index < endIndex2; index++)
    {
        factor += dataA2[index] * dataA2[index];
    }
    factor *= coeff;
    daal::internal::Math<AlgorithmFPType,cpu>::vExp(1, &factor, dataR);
}

template <typename AlgorithmFPType, CpuType cpu>
void KernelImplRBF<fastCSR, AlgorithmFPType, cpu>::computeInternalMatrixVector(
            size_t nFeatures,
            size_t nVectors1, const AlgorithmFPType *dataA1, const size_t *colIndicesA1, const size_t *rowOffsetsA1,
            size_t nVectors2, const AlgorithmFPType *dataA2, const size_t *colIndicesA2, const size_t *rowOffsetsA2,
            AlgorithmFPType *dataR, const ParameterBase *par)
{
    const Parameter *rbfPar = static_cast<const Parameter *>(par);
    AlgorithmFPType coeff = (AlgorithmFPType)(-0.5 / (rbfPar->sigma * rbfPar->sigma));
    size_t startIndex2 = rowOffsetsA2[0] - 1;
    size_t endIndex2   = rowOffsetsA2[1] - 1;

    AlgorithmFPType factor = 0.0;
    for (size_t index = startIndex2; index < endIndex2; index++)
    {
        factor += dataA2[index] * dataA2[index];
    }
    for (size_t i = 0; i < nVectors1; i++)
    {
        size_t startIndex1 = rowOffsetsA1[i]   - 1;
        size_t endIndex1   = rowOffsetsA1[i+1] - 1;
        dataR[i] = computeDotProduct(startIndex1, endIndex1, dataA1, colIndicesA1,
                                     startIndex2, endIndex2, dataA2, colIndicesA2);
        dataR[i] = -2.0 * dataR[i] + factor;
        for (size_t index = startIndex1; index < endIndex1; index++)
        {
            dataR[i] += dataA1[index] * dataA1[index];
        }
        dataR[i] *= coeff;
    }
    daal::internal::Math<AlgorithmFPType,cpu>::vExp(nVectors1, dataR, dataR);
}

template <typename AlgorithmFPType, CpuType cpu>
void KernelImplRBF<fastCSR, AlgorithmFPType, cpu>::computeInternalMatrixMatrix(
            size_t nFeatures,
            size_t nVectors1, const AlgorithmFPType *dataA1, const size_t *colIndicesA1, const size_t *rowOffsetsA1,
            size_t nVectors2, const AlgorithmFPType *dataA2, const size_t *colIndicesA2, const size_t *rowOffsetsA2,
            AlgorithmFPType *dataR, const ParameterBase *par, bool inputTablesSame)
{
    const Parameter *rbfPar = static_cast<const Parameter *>(par);
    const AlgorithmFPType coeff = (AlgorithmFPType)(-0.5 / (rbfPar->sigma * rbfPar->sigma));
    const AlgorithmFPType zero = 0.0;
    const AlgorithmFPType negTwo = -2.0;

    if (inputTablesSame)
    {
        for (size_t i = 0; i < nVectors1; i++)
        {
            for (size_t j = 0; j <= i; j++)
            {
                dataR[i*nVectors1 + j] = computeDotProduct(rowOffsetsA1[i] - 1, rowOffsetsA1[i+1] - 1, dataA1, colIndicesA1,
                                                           rowOffsetsA1[j] - 1, rowOffsetsA1[j+1] - 1, dataA1, colIndicesA1);
            }
        }
        for (size_t i = 0; i < nVectors1; i++)
        {
            for (size_t k = 0; k < i; k++)
            {
                dataR[i * nVectors1 + k] = coeff * (dataR[i * nVectors1 + i] + dataR[k * nVectors1 + k] +
                        negTwo * dataR[i * nVectors1 + k]);
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
    else
    {
        for (size_t i = 0; i < nVectors1; i++)
        {
            for (size_t j = 0; j < nVectors2; j++)
            {
                dataR[i*nVectors2 + j] = computeDotProduct(rowOffsetsA1[i] - 1, rowOffsetsA1[i+1] - 1, dataA1, colIndicesA1,
                                                           rowOffsetsA2[j] - 1, rowOffsetsA2[j+1] - 1, dataA2, colIndicesA2);
            }
        }
        AlgorithmFPType *buffer = (AlgorithmFPType *)daal::services::daal_malloc((nVectors1 + nVectors2) * sizeof(AlgorithmFPType));
        if (!buffer) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }
        AlgorithmFPType *sqrDataA1 = buffer;
        AlgorithmFPType *sqrDataA2 = buffer + nVectors1;
        for (size_t i = 0; i < nVectors1; i++)
        {
            sqrDataA1[i] = zero;
            for (size_t j = rowOffsetsA1[i] - 1; j < rowOffsetsA1[i + 1] - 1; j++)
            {
                sqrDataA1[i] += dataA1[j] * dataA1[j];
            }
        }
        for (size_t i = 0; i < nVectors2; i++)
        {
            sqrDataA2[i] = zero;
            for (size_t j = rowOffsetsA2[i] - 1; j < rowOffsetsA2[i + 1] - 1; j++)
            {
                sqrDataA2[i] += dataA2[j] * dataA2[j];
            }
        }
        for (size_t i = 0; i < nVectors1; i++)
        {
            for (size_t k = 0; k < nVectors2; k++)
            {
                dataR[i * nVectors2 + k] *= negTwo;
                dataR[i * nVectors2 + k] += (sqrDataA1[i] + sqrDataA2[k]);
                dataR[i * nVectors2 + k] *= coeff;
            }
        }
        daal::internal::Math<AlgorithmFPType,cpu>::vExp(nVectors1 * nVectors2, dataR, dataR);
        daal::services::daal_free(buffer);
    }
}

} // namespace internal

} // namespace rbf

} // namespace kernel_function

} // namespace algorithms

} // namespace daal


#endif
