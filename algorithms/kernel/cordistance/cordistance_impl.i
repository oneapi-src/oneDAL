/* file: cordistance_impl.i */
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
//  Implementation of distances
//--
*/

#include "service_micro_table.h"
#include "service_math.h"
#include "service_memory.h"
#include "service_blas.h"
#include "service_numeric_table.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace correlation_distance
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu> bool isFull(NumericTableIface::StorageLayout layout);
template <typename algorithmFPType, CpuType cpu> bool isUpper(NumericTableIface::StorageLayout layout);
template <typename algorithmFPType, CpuType cpu> bool isLower(NumericTableIface::StorageLayout layout);
/**
 *  \brief Kernel for Correlation distances calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void DistanceKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                           const size_t nr, NumericTable *r[],
                                                           const daal::algorithms::Parameter *par)
{
    NumericTable *xTable = const_cast<NumericTable *>( a[0] );  /* Input data */
    MKL_INT n   = (MKL_INT)(xTable->getNumberOfRows());         /* Number of input feature vectors */
    MKL_INT dim = (MKL_INT)(xTable->getNumberOfColumns());      /* Dimension of input feature vectors */
    MKL_INT ione = 1;
    const algorithmFPType one = (algorithmFPType)1.0;

    SmartPtr<cpu> aXXT(n * n * sizeof(algorithmFPType)); /* Buffer for algorithmFPTypeediate results */
    SmartPtr<cpu> aVOne(dim * sizeof(algorithmFPType));  /* Vector of ones */
    SmartPtr<cpu> aXSum(n * sizeof(algorithmFPType));    /* Vector of sums of rows of matrix X */
    algorithmFPType *xxt = (algorithmFPType *)aXXT.get();
    algorithmFPType *vone = (algorithmFPType *)aVOne.get();
    algorithmFPType *xsum = (algorithmFPType *)aXSum.get();
    if(!xxt || !vone || !xsum) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }
    for(size_t i = 0; i < dim; i++)
        vone[i] = one;

    const NumericTableIface::StorageLayout rLayout = r[0]->getDataLayout();

    {
        ReadRows<algorithmFPType, cpu> xTableBD(*xTable, 0, n);
        const algorithmFPType *x = xTableBD.get();      /* Input data           */
        /* Calculate X*X' */
        char uplo  = 'U';
        char trans = 'T';
        algorithmFPType alpha = 1.0;
        algorithmFPType beta = 0.0;
        Blas<algorithmFPType, cpu>::xsyrk(&uplo, &trans, &n, &dim, &alpha, const_cast<algorithmFPType*>(x), &dim, &beta, xxt, &n);
        /* Calculate X*vone */
        trans = 'T';
        alpha = 1.0;
        beta  = 0.0;
        Blas<algorithmFPType, cpu>::xgemv(&trans, &dim, &n, &alpha, const_cast<algorithmFPType*>(x), &dim, vone, &ione, &beta, xsum, &ione);
    }

    algorithmFPType invDim = one / (algorithmFPType)dim;
    for (size_t i = 0; i < n; i++)
    {
        if (xxt[i * n + i] != 0.0)
        {
            xxt[i * n + i] = one / daal::internal::Math<algorithmFPType,cpu>::sSqrt(xxt[i * n + i] - xsum[i] * xsum[i] * invDim);
        }
    }

    if(isFull<algorithmFPType, cpu>(rLayout))
    {
        WriteOnlyRows<algorithmFPType, cpu> resBD(*r[0], 0, n);
        algorithmFPType *d = resBD.get();      /* Resulting distances  */

        /* Pack the results into output array */
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                d[i * n + j] = one - (xxt[i * n + j] - xsum[i] * xsum[j] * invDim) *
                               xxt[i * n + i] * xxt[j * n + j];
                d[j * n + i] = d[i * n + j];
            }
            d[i * n + i] = one;
        }
    }
    else
    {
        PackedArrayMicroTable<algorithmFPType, writeOnly, cpu> rPackedMicroTable(r[0]);
        algorithmFPType *d = nullptr;      /* Resulting distances  */
        rPackedMicroTable.getPackedArray(&d);

        /* Pack the results into output array */
        size_t dIndex = 0;
        if(isLower<algorithmFPType, cpu>(rLayout))
        {
            for (size_t i = 0; i < n; i++)
            {
                for (size_t j = 0; j < i; j++)
                {
                    d[dIndex++] = one - (xxt[i * n + j] - xsum[i] * xsum[j] * invDim) *
                                  xxt[i * n + i] * xxt[j * n + j];
                }
                d[dIndex++] = one;
            }
        }
        else if(isUpper<algorithmFPType, cpu>(rLayout))
        {
            for (size_t j = 0; j < n; j++)
            {
                d[dIndex++] = one;
                for (size_t i = j + 1; i < n; i++)
                {
                    d[dIndex++] = one - (xxt[i * n + j] - xsum[i] * xsum[j] * invDim) *
                                  xxt[i * n + i] * xxt[j * n + j];
                }
            }
        }
        else
        {
            rPackedMicroTable.release();
            this->_errors->add(services::ErrorIncorrectTypeOfOutputNumericTable); return;
        }
        rPackedMicroTable.release();
    }
}

template <typename algorithmFPType, CpuType cpu>
bool isFull(NumericTableIface::StorageLayout layout)
{
    int layoutInt = (int) layout;
    if (packed_mask & layoutInt)
    {
        return false;
    }
    return true;
}

template <typename algorithmFPType, CpuType cpu>
bool isUpper(NumericTableIface::StorageLayout layout)
{
    if (layout == NumericTableIface::upperPackedSymmetricMatrix  ||
        layout == NumericTableIface::upperPackedTriangularMatrix)
    {
        return true;
    }
    return false;
}

template <typename algorithmFPType, CpuType cpu>
bool isLower(NumericTableIface::StorageLayout layout)
{
    if (layout == NumericTableIface::lowerPackedSymmetricMatrix  ||
        layout == NumericTableIface::lowerPackedTriangularMatrix)
    {
        return true;
    }
    return false;
}

} // namespace internal

} // namespace correlation_distance

} // namespace algorithms

} // namespace daal
