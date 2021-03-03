/* file: kernel_function_polynomial_dense_default_impl.i */
/*******************************************************************************
* Copyright 2021 Intel Corporation
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
//  Polynomial kernel functions implementation
//--
*/

#ifndef __KERNEL_FUNCTION_POLYNOMIAL_DENSE_DEFAULT_IMPL_I__
#define __KERNEL_FUNCTION_POLYNOMIAL_DENSE_DEFAULT_IMPL_I__

#include "src/algorithms/kernel_function/polynomial/kernel_function_types_polynomial.h"

#include "src/externals/service_blas.h"
#include "src/externals/service_stat.h"
#include "src/threading/threading.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_math.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace polynomial
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplPolynomial<defaultDense, algorithmFPType, cpu>::computeInternalVectorVector(const NumericTable * a1,
                                                                                                       const NumericTable * a2, NumericTable * r,
                                                                                                       const ParameterBase * par)
{
    return services::ErrorMethodNotImplemented;
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplPolynomial<defaultDense, algorithmFPType, cpu>::computeInternalMatrixVector(const NumericTable * a1,
                                                                                                       const NumericTable * a2, NumericTable * r,
                                                                                                       const ParameterBase * par)
{
    return services::ErrorMethodNotImplemented;
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplPolynomial<defaultDense, algorithmFPType, cpu>::computeInternalMatrixMatrix(const NumericTable * a1,
                                                                                                       const NumericTable * a2, NumericTable * r,
                                                                                                       const ParameterBase * par)
{
    SafeStatus safeStat;

    char trans   = 'T';
    char notrans = 'N';

    const size_t nFeatures = a1->getNumberOfColumns();
    const size_t nVectors1 = a1->getNumberOfRows();
    const size_t nVectors2 = a2->getNumberOfRows();

    const Parameter * polyPar = static_cast<const Parameter *>(par);
    algorithmFPType alpha     = (algorithmFPType)(polyPar->scale);
    algorithmFPType beta      = 0.0;
    algorithmFPType shift     = (algorithmFPType)(polyPar->shift);
    algorithmFPType degree    = (algorithmFPType)(polyPar->degree);

    const services::Status retStat =
        Blas<algorithmFPType, cpu>::xgemm_blocked(&trans, &notrans, (DAAL_INT *)&nVectors2, (DAAL_INT *)&nVectors1, (DAAL_INT *)&nFeatures, &alpha,
                                                  a2, (DAAL_INT *)&nFeatures, a1, (DAAL_INT *)&nFeatures, &beta, r, (DAAL_INT *)&nVectors2, 128, 128);
    if (!retStat) return retStat;

    if (shift != 0.0 || degree > 1.0)
    {
        const size_t blockSize = 256;
        const size_t nBlocks   = nVectors1 / blockSize + !!(nVectors1 % blockSize);

        daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            const size_t startRow     = iBlock * blockSize;
            const size_t finishRow    = iBlock < nBlocks - 1 ? (iBlock + 1) * blockSize : nVectors1;
            const size_t nRowsInBlock = finishRow - startRow;

            WriteRows<algorithmFPType, cpu> mtR(r, startRow, nRowsInBlock);
            DAAL_CHECK_MALLOC_THR(mtR.get());
            algorithmFPType * dataR = mtR.get();

            if (shift != 0.0)
            {
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = 0; i < nRowsInBlock * nVectors2; ++i)
                {
                    dataR[i] += shift;
                }
            }

            if (degree > 1.0)
            {
                Math<algorithmFPType, cpu>::vPowx(nRowsInBlock * nVectors2, dataR, degree, dataR);
            }
        });
    }

    return safeStat.detach();
}

} // namespace internal
} // namespace polynomial
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif
