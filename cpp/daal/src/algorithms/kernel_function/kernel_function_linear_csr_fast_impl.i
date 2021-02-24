/* file: kernel_function_linear_csr_fast_impl.i */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

#ifndef __KERNEL_FUNCTION_LINEAR_CSR_FAST_IMPL_I__
#define __KERNEL_FUNCTION_LINEAR_CSR_FAST_IMPL_I__

#include "algorithms/kernel_function/kernel_function_types_linear.h"
#include "src/algorithms/kernel_function/kernel_function_csr_impl.i"

#include "src/threading/threading.h"
#include "src/externals/service_spblas.h"
#include "src/externals/service_blas.h"

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
services::Status KernelImplLinear<fastCSR, algorithmFPType, cpu>::computeInternalVectorVector(const NumericTable * a1, const NumericTable * a2,
                                                                                              NumericTable * r, const ParameterBase * par)
{
    //prepareData
    ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), par->rowIndexX, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const size_t * rowOffsetsA1 = mtA1.rows();

    ReadRowsCSR<algorithmFPType, cpu> mtA2(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a2)), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const size_t * rowOffsetsA2 = mtA2.rows();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, par->rowIndexResult, 1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    const Parameter * linPar = static_cast<const Parameter *>(par);

    //compute
    dataR[0] = computeDotProduct(rowOffsetsA1[0] - 1, rowOffsetsA1[1] - 1, mtA1.values(), mtA1.cols(), rowOffsetsA2[0] - 1, rowOffsetsA2[1] - 1,
                                 mtA2.values(), mtA2.cols());
    dataR[0] = dataR[0] * linPar->k + linPar->b;

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplLinear<fastCSR, algorithmFPType, cpu>::computeInternalMatrixVector(const NumericTable * a1, const NumericTable * a2,
                                                                                              NumericTable * r, const ParameterBase * par)
{
    //prepareData
    const size_t nVectors1 = a1->getNumberOfRows();

    ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const size_t * rowOffsetsA1 = mtA1.rows();

    ReadRowsCSR<algorithmFPType, cpu> mtA2(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a2)), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const size_t * rowOffsetsA2 = mtA2.rows();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    const Parameter * linPar = static_cast<const Parameter *>(par);
    algorithmFPType b        = (algorithmFPType)(linPar->b);
    algorithmFPType k        = (algorithmFPType)(linPar->k);

    //compute
    for (size_t i = 0; i < nVectors1; i++)
    {
        dataR[i] = computeDotProduct(rowOffsetsA1[i] - 1, rowOffsetsA1[i + 1] - 1, mtA1.values(), mtA1.cols(), rowOffsetsA2[0] - 1,
                                     rowOffsetsA2[1] - 1, mtA2.values(), mtA2.cols());
        dataR[i] = dataR[i] * k + b;
    }

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplLinear<fastCSR, algorithmFPType, cpu>::computeInternalMatrixMatrix(const NumericTable * a1, const NumericTable * a2,
                                                                                              NumericTable * r, const ParameterBase * par)
{
    //prepareData
    const size_t nVectors1 = a1->getNumberOfRows();
    const size_t nVectors2 = a2->getNumberOfRows();
    const size_t nFeatures = a1->getNumberOfColumns();

    const Parameter * linPar = static_cast<const Parameter *>(par);
    algorithmFPType b        = (algorithmFPType)(linPar->b);
    algorithmFPType k        = (algorithmFPType)(linPar->k);

    const size_t blockSize = 256;
    const size_t nBlocks1  = nVectors1 / blockSize + !!(nVectors1 % blockSize);
    const size_t nBlocks2  = nVectors2 / blockSize + !!(nVectors2 % blockSize);

    const bool isSOARes = r->getDataLayout() & NumericTableIface::soa;

    TlsMem<algorithmFPType, cpu> tlsMklBuff(blockSize * blockSize);
    SafeStatus safeStat;
    daal::conditional_threader_for((nVectors1 > 512), nBlocks1, [&, isSOARes](const size_t iBlock1) {
        const size_t nRowsInBlock1 = (iBlock1 != nBlocks1 - 1) ? blockSize : nVectors1 - iBlock1 * blockSize;
        const size_t startRow1     = iBlock1 * blockSize;

        ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), startRow1, nRowsInBlock1, true);
        DAAL_CHECK_BLOCK_STATUS_THR(mtA1);
        const algorithmFPType * dataA1 = mtA1.values();
        const size_t * colIndicesA1    = mtA1.cols();
        const size_t * rowOffsetsA1    = mtA1.rows();

        WriteOnlyRows<algorithmFPType, cpu> mtRRows;
        if (!isSOARes)
        {
            mtRRows.set(r, startRow1, nRowsInBlock1);
            DAAL_CHECK_MALLOC_THR(mtRRows.get());
        }
        daal::conditional_threader_for((nVectors2 > 512), nBlocks2, [&, nVectors2, nBlocks2](const size_t iBlock2) {
            const size_t nRowsInBlock2 = (iBlock2 != nBlocks2 - 1) ? blockSize : nVectors2 - iBlock2 * blockSize;
            const size_t startRow2     = iBlock2 * blockSize;

            ReadRowsCSR<algorithmFPType, cpu> mtA2(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a2)), startRow2, nRowsInBlock2,
                                                   true);
            DAAL_CHECK_BLOCK_STATUS_THR(mtA2);
            const algorithmFPType * dataA2 = mtA2.values();
            const size_t * colIndicesA2    = mtA2.cols();
            const size_t * rowOffsetsA2    = mtA2.rows();

            if (!isSOARes)
            {
                const size_t ldc              = nVectors2;
                algorithmFPType * const dataR = mtRRows.get();
                SpBlas<algorithmFPType, cpu>::xxgemm_a_bt(dataA1, colIndicesA1, rowOffsetsA1, dataA2, colIndicesA2, rowOffsetsA2, nRowsInBlock1,
                                                          nRowsInBlock2, nFeatures, dataR + startRow2, ldc);

                if (k != (algorithmFPType)1.0 || b != (algorithmFPType)0.0)
                {
                    for (size_t i = 0; i < nRowsInBlock1; i++)
                    {
                        for (size_t j = 0; j < nRowsInBlock2; j++)
                        {
                            dataR[i * ldc + j + startRow2] = dataR[i * ldc + j + startRow2] * k + b;
                        }
                    }
                }
            }
            else
            {
                const size_t ldc                = blockSize;
                algorithmFPType * const mklBuff = tlsMklBuff.local();

                SpBlas<algorithmFPType, cpu>::xxgemm_a_bt(dataA2, colIndicesA2, rowOffsetsA2, dataA1, colIndicesA1, rowOffsetsA1, nRowsInBlock2,
                                                          nRowsInBlock1, nFeatures, mklBuff, ldc);

                if (k != (algorithmFPType)1.0 || b != (algorithmFPType)0.0)
                {
                    for (size_t i = 0; i < nRowsInBlock2; i++)
                    {
                        for (size_t j = 0; j < nRowsInBlock1; j++)
                        {
                            mklBuff[i * ldc + j] = mklBuff[i * ldc + j] * k + b;
                        }
                    }
                }

                for (size_t j = 0; j < nRowsInBlock2; ++j)
                {
                    WriteOnlyColumns<algorithmFPType, cpu> mtRColumns(r, startRow2 + j, startRow1, nRowsInBlock1);
                    DAAL_CHECK_BLOCK_STATUS_THR(mtRColumns);
                    algorithmFPType * const dataRBlock   = mtRColumns.get();
                    algorithmFPType * const mklBuffBlock = &mklBuff[j * blockSize];
                    internal::Helper<algorithmFPType, cpu>::copy(dataRBlock, mklBuffBlock, nRowsInBlock1);
                }
            }
        });
    });

    return services::Status();
}

} // namespace internal
} // namespace linear
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif
