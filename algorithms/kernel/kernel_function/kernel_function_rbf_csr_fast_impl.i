/* file: kernel_function_rbf_csr_fast_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include "service_math.h"
#include "service_numeric_table.h"
#include "kernel_function_csr_impl.i"
#include "threading.h"

#include "service_spblas.h"

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
services::Status KernelImplRBF<fastCSR, algorithmFPType, cpu>::computeInternalVectorVector(const NumericTable * a1, const NumericTable * a2,
                                                                                           NumericTable * r, const ParameterBase * par)
{
    //prepareData
    ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), par->rowIndexX, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const algorithmFPType * dataA1 = mtA1.values();
    const size_t * rowOffsetsA1    = mtA1.rows();

    ReadRowsCSR<algorithmFPType, cpu> mtA2(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a2)), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const algorithmFPType * dataA2 = mtA2.values();
    const size_t * rowOffsetsA2    = mtA2.rows();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, par->rowIndexResult, 1);
    DAAL_CHECK_BLOCK_STATUS(mtR);

    //compute
    const Parameter * rbfPar    = static_cast<const Parameter *>(par);
    const algorithmFPType coeff = (algorithmFPType)(-0.5 / (rbfPar->sigma * rbfPar->sigma));
    const size_t startIndex1    = rowOffsetsA1[0] - 1;
    const size_t startIndex2    = rowOffsetsA2[0] - 1;
    const size_t endIndex1      = rowOffsetsA1[1] - 1;
    const size_t endIndex2      = rowOffsetsA2[1] - 1;
    algorithmFPType factor      = computeDotProduct(startIndex1, endIndex1, dataA1, mtA1.cols(), startIndex2, endIndex2, dataA2, mtA2.cols());
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
    daal::internal::Math<algorithmFPType, cpu>::vExp(1, &factor, mtR.get());

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplRBF<fastCSR, algorithmFPType, cpu>::computeInternalMatrixVector(const NumericTable * a1, const NumericTable * a2,
                                                                                           NumericTable * r, const ParameterBase * par)
{
    //prepareData
    const size_t nVectors1 = a1->getNumberOfRows();

    ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const algorithmFPType * dataA1 = mtA1.values();
    const size_t * rowOffsetsA1    = mtA1.rows();

    ReadRowsCSR<algorithmFPType, cpu> mtA2(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a2)), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const algorithmFPType * dataA2 = mtA2.values();
    const size_t * rowOffsetsA2    = mtA2.rows();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, par->rowIndexResult, 1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    //compute
    const Parameter * rbfPar    = static_cast<const Parameter *>(par);
    const algorithmFPType coeff = (algorithmFPType)(-0.5 / (rbfPar->sigma * rbfPar->sigma));
    const size_t startIndex2    = rowOffsetsA2[0] - 1;
    const size_t endIndex2      = rowOffsetsA2[1] - 1;

    algorithmFPType factor = 0.0;
    for (size_t index = startIndex2; index < endIndex2; index++)
    {
        factor += dataA2[index] * dataA2[index];
    }
    for (size_t i = 0; i < nVectors1; i++)
    {
        size_t startIndex1 = rowOffsetsA1[i] - 1;
        size_t endIndex1   = rowOffsetsA1[i + 1] - 1;
        dataR[i]           = computeDotProduct(startIndex1, endIndex1, dataA1, mtA1.cols(), startIndex2, endIndex2, dataA2, mtA2.cols());
        dataR[i]           = -2.0 * dataR[i] + factor;
        for (size_t index = startIndex1; index < endIndex1; index++)
        {
            dataR[i] += dataA1[index] * dataA1[index];
        }
        dataR[i] *= coeff;

        // make all values less than threshold as threshold value
        // to fix slow work on vExp on large negative inputs
        if (dataR[i] < Math<algorithmFPType, cpu>::vExpThreshold())
        {
            dataR[i] = Math<algorithmFPType, cpu>::vExpThreshold();
        }
    }
    daal::internal::Math<algorithmFPType, cpu>::vExp(nVectors1, dataR, dataR);

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplRBF<fastCSR, algorithmFPType, cpu>::computeInternalMatrixMatrix(const NumericTable * a1, const NumericTable * a2,
                                                                                           NumericTable * r, const ParameterBase * par)
{
    //prepareData
    const size_t nVectors1 = a1->getNumberOfRows();
    const size_t nVectors2 = a2->getNumberOfRows();

    ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const algorithmFPType * dataA1 = mtA1.values();
    const size_t * colIndicesA1    = mtA1.cols();
    const size_t * rowOffsetsA1    = mtA1.rows();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    //compute
    const Parameter * rbfPar     = static_cast<const Parameter *>(par);
    const algorithmFPType coeff  = (algorithmFPType)(-0.5 / (rbfPar->sigma * rbfPar->sigma));
    const algorithmFPType zero   = 0.0;
    const algorithmFPType negTwo = -2.0;

    if (a1 == a2)
    {
        SpBlas<algorithmFPType, cpu>::xsyrk_a_at(dataA1, colIndicesA1, rowOffsetsA1, nVectors1, a1->getNumberOfColumns(), dataR);

        daal::threader_for_optional(nVectors1, nVectors1, [=](size_t i) {
            for (size_t k = 0; k < i; k++)
            {
                dataR[i * nVectors1 + k] = coeff * (dataR[i * nVectors1 + i] + dataR[k * nVectors1 + k] + negTwo * dataR[i * nVectors1 + k]);
            }
        });
        daal::threader_for_optional(nVectors1, nVectors1, [=](size_t i) {
            dataR[i * nVectors1 + i] = zero;
            daal::internal::Math<algorithmFPType, cpu>::vExp(i + 1, dataR + i * nVectors1, dataR + i * nVectors1);
        });
        daal::threader_for_optional(nVectors1, nVectors1, [=](size_t i) {
            for (size_t k = i + 1; k < nVectors1; k++)
            {
                dataR[i * nVectors1 + k] = dataR[k * nVectors1 + i];
            }
        });
    }
    else
    {
        ReadRowsCSR<algorithmFPType, cpu> mtA2(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a2)), 0, nVectors2);
        DAAL_CHECK_BLOCK_STATUS(mtA2);
        const algorithmFPType * dataA2 = mtA2.values();
        const size_t * colIndicesA2    = mtA2.cols();
        const size_t * rowOffsetsA2    = mtA2.rows();

        DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, nVectors1, nVectors2);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nVectors1 + nVectors2, sizeof(algorithmFPType));

        daal::internal::TArray<algorithmFPType, cpu> aBuf((nVectors1 + nVectors2));
        DAAL_CHECK(aBuf.get(), services::ErrorMemoryAllocationFailed);
        algorithmFPType * buffer = aBuf.get();

        algorithmFPType * sqrDataA1 = buffer;
        algorithmFPType * sqrDataA2 = buffer + nVectors1;

        SpBlas<algorithmFPType, cpu>::xgemm_a_bt(dataA1, colIndicesA1, rowOffsetsA1, dataA2, colIndicesA2, rowOffsetsA2, nVectors1, nVectors2,
                                                 a1->getNumberOfColumns(), dataR);

        daal::threader_for_optional(nVectors1, nVectors1, [=](size_t i) {
            sqrDataA1[i] = zero;
            for (size_t j = rowOffsetsA1[i] - 1; j < rowOffsetsA1[i + 1] - 1; j++)
            {
                sqrDataA1[i] += dataA1[j] * dataA1[j];
            }
        });
        daal::threader_for_optional(nVectors2, nVectors2, [=](size_t i) {
            sqrDataA2[i] = zero;
            for (size_t j = rowOffsetsA2[i] - 1; j < rowOffsetsA2[i + 1] - 1; j++)
            {
                sqrDataA2[i] += dataA2[j] * dataA2[j];
            }
        });
        daal::threader_for_optional(nVectors1, nVectors1, [=](size_t i) {
            for (size_t k = 0; k < nVectors2; k++)
            {
                dataR[i * nVectors2 + k] *= negTwo;
                dataR[i * nVectors2 + k] += (sqrDataA1[i] + sqrDataA2[k]);
                dataR[i * nVectors2 + k] *= coeff;

                // make all values less than threshold as threshold value
                // to fix slow work on vExp on large negative inputs
                if (dataR[i * nVectors2 + k] < Math<algorithmFPType, cpu>::vExpThreshold())
                {
                    dataR[i * nVectors2 + k] = Math<algorithmFPType, cpu>::vExpThreshold();
                }
            }
        });

        daal::internal::Math<algorithmFPType, cpu>::vExp(nVectors1 * nVectors2, dataR, dataR);
    }
    return services::Status();
}

} // namespace internal

} // namespace rbf

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif
