/* file: kernel_function_linear_dense_default_impl.i */
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

#ifndef __KERNEL_FUNCTION_LINEAR_DENSE_DEFAULT_IMPL_I__
#define __KERNEL_FUNCTION_LINEAR_DENSE_DEFAULT_IMPL_I__

#include "kernel_function_types_linear.h"

#include "service_blas.h"
#include "service_stat.h"
#include "threading.h"
#include "service_error_handling.h"

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
services::Status KernelImplLinear<defaultDense, algorithmFPType, cpu>::computeInternalVectorVector(const NumericTable * a1, const NumericTable * a2,
                                                                                                   NumericTable * r, const ParameterBase * par)
{
    //prepareData
    const size_t nFeatures = a1->getNumberOfColumns();

    ReadRows<algorithmFPType, cpu> mtA1(*const_cast<NumericTable *>(a1), par->rowIndexX, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const algorithmFPType * dataA1 = mtA1.get();

    ReadRows<algorithmFPType, cpu> mtA2(*const_cast<NumericTable *>(a2), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const algorithmFPType * dataA2 = mtA2.get();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, par->rowIndexResult, 1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    //compute
    const Parameter * linPar = static_cast<const Parameter *>(par);
    dataR[0]                 = 0.0;
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nFeatures; i++)
    {
        dataR[0] += dataA1[i] * dataA2[i];
    }
    dataR[0] = dataR[0] * linPar->k + linPar->b;

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplLinear<defaultDense, algorithmFPType, cpu>::computeInternalMatrixVector(const NumericTable * a1, const NumericTable * a2,
                                                                                                   NumericTable * r, const ParameterBase * par)
{
    //prepareData
    const size_t nVectors1 = a1->getNumberOfRows();
    const size_t nFeatures = a1->getNumberOfColumns();

    ReadRows<algorithmFPType, cpu> mtA1(*const_cast<NumericTable *>(a1), 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const algorithmFPType * dataA1 = mtA1.get();

    ReadRows<algorithmFPType, cpu> mtA2(*const_cast<NumericTable *>(a2), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const algorithmFPType * dataA2 = mtA2.get();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, par->rowIndexResult, 1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    //compute
    const Parameter * linPar = static_cast<const Parameter *>(par);
    algorithmFPType b        = (algorithmFPType)(linPar->b);
    algorithmFPType k        = (algorithmFPType)(linPar->k);
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

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplLinear<defaultDense, algorithmFPType, cpu>::computeInternalMatrixMatrix(const NumericTable * a1, const NumericTable * a2,
                                                                                                   NumericTable * r, const ParameterBase * par)
{
    SafeStatus safeStat;

    const char trans   = 'T';
    const char notrans = 'N';

    const size_t nFeatures = a1->getNumberOfColumns();
    const size_t nVectors1 = a1->getNumberOfRows();
    const size_t nVectors2 = a2->getNumberOfRows();

    const Parameter * linPar = static_cast<const Parameter *>(par);
    algorithmFPType alpha    = (algorithmFPType)(linPar->k);
    algorithmFPType beta     = 0.0;
    algorithmFPType b        = (algorithmFPType)(linPar->b);

    if (a1 != a2)
    {
        ReadRows<algorithmFPType, cpu> mtA1(*const_cast<NumericTable *>(a1), 0, nVectors1);
        DAAL_CHECK_BLOCK_STATUS(mtA1);
        const algorithmFPType * dataA1 = const_cast<algorithmFPType *>(mtA1.get());

        ReadRows<algorithmFPType, cpu> mtA2(*const_cast<NumericTable *>(a2), 0, nVectors2);
        DAAL_CHECK_BLOCK_STATUS(mtA2);
        const algorithmFPType * dataA2 = const_cast<algorithmFPType *>(mtA2.get());

        WriteOnlyRows<algorithmFPType, cpu> mtR(r, 0, nVectors1);
        DAAL_CHECK_BLOCK_STATUS(mtR);
        algorithmFPType * dataR = mtR.get();

        Blas<algorithmFPType, cpu>::xgemm(&trans, &notrans, (DAAL_INT *)&nVectors2, (DAAL_INT *)&nVectors1, (DAAL_INT *)&nFeatures, &alpha,
                                          (algorithmFPType *)dataA2, (DAAL_INT *)&nFeatures, (algorithmFPType *)dataA1, (DAAL_INT *)&nFeatures, &beta,
                                          dataR, (DAAL_INT *)&nVectors2);
    }
    else
    {
        services::Status retStat = Blas<algorithmFPType, cpu>::xgemm_blocked(&trans, &notrans, (DAAL_INT *)&nVectors2, (DAAL_INT *)&nVectors1,
                                                                             (DAAL_INT *)&nFeatures, &alpha, a2, (DAAL_INT *)&nFeatures, a1,
                                                                             (DAAL_INT *)&nFeatures, &beta, r, (DAAL_INT *)&nVectors2);
        if (!retStat) return retStat;
    }

    if (b != 0.0)
    {
        WriteOnlyRows<algorithmFPType, cpu> mtR(r, 0, nVectors1);
        DAAL_CHECK_BLOCK_STATUS(mtR);
        algorithmFPType * dataR = mtR.get();

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nVectors1 * nVectors2; i++)
        {
            dataR[i] = dataR[i] + b;
        }
    }

    return safeStat.detach();
}

} // namespace internal

} // namespace linear

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif
