/* file: kernel_function_linear_csr_fast_impl.i */
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
//  Linear kernel functions implementation
//--
*/

#ifndef __KERNEL_FUNCTION_LINEAR_CSR_FAST_IMPL_I__
#define __KERNEL_FUNCTION_LINEAR_CSR_FAST_IMPL_I__

#include "kernel_function_types_linear.h"
#include "kernel_function_csr_impl.i"

#include "threading.h"
#include "service_spblas.h"

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

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, par->rowIndexResult, 1);
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

    ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const algorithmFPType * dataA1 = mtA1.values();
    const size_t * colIndicesA1    = mtA1.cols();
    const size_t * rowOffsetsA1    = mtA1.rows();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    const Parameter * linPar = static_cast<const Parameter *>(par);
    algorithmFPType b        = (algorithmFPType)(linPar->b);
    algorithmFPType k        = (algorithmFPType)(linPar->k);

    //compute
    if (a1 == a2)
    {
        SpBlas<algorithmFPType, cpu>::xsyrk_a_at(dataA1, colIndicesA1, rowOffsetsA1, nVectors1, a1->getNumberOfColumns(), dataR);

        if (k != (algorithmFPType)1.0 || b != (algorithmFPType)0.0)
        {
            daal::threader_for_optional(nVectors1, nVectors1, [=](size_t i) {
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j <= i; j++)
                {
                    dataR[i * nVectors1 + j] = dataR[i * nVectors1 + j] * k + b;
                }
            });
        }

        daal::threader_for_optional(nVectors1, nVectors1, [=](size_t i) {
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = i + 1; j < nVectors1; j++)
            {
                dataR[i * nVectors1 + j] = dataR[j * nVectors1 + i];
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

        SpBlas<algorithmFPType, cpu>::xgemm_a_bt(dataA1, colIndicesA1, rowOffsetsA1, dataA2, colIndicesA2, rowOffsetsA2, nVectors1, nVectors2,
                                                 a1->getNumberOfColumns(), dataR);

        if (k != (algorithmFPType)1.0 || b != (algorithmFPType)0.0)
        {
            daal::threader_for_optional(nVectors1, nVectors1, [=](size_t i) {
                for (size_t j = 0; j < nVectors2; j++)
                {
                    dataR[i * nVectors2 + j] = dataR[i * nVectors2 + j] * k + b;
                }
            });
        }
    }

    return services::Status();
}

} // namespace internal

} // namespace linear

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif
