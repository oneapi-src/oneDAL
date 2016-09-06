/* file: cosdistance_batch_impl.i */
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
#include "daal_defines.h"
#include "service_blas.h"
#include "threading.h"

static const int blockSizeDefault=128;
#include "cosdistance_full_impl.i"
#include "cosdistance_up_impl.i"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace cosine_distance
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu> bool isFull(NumericTableIface::StorageLayout layout);
template <typename algorithmFPType, CpuType cpu> bool isUpper(NumericTableIface::StorageLayout layout);
template <typename algorithmFPType, CpuType cpu> bool isLower(NumericTableIface::StorageLayout layout);

/**
 *  \brief Kernel for Cosine distances calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void DistanceKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                           const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    NumericTable *xTable = const_cast<NumericTable *>( a[0] );  /* Input data */
    NumericTable *rTable = const_cast<NumericTable *>( r[0] );  /* Output data */
    MKL_INT n   = (MKL_INT)(xTable->getNumberOfRows());         /* Number of input feature vectors */
    MKL_INT p   = (MKL_INT)(xTable->getNumberOfColumns());      /* Number of input vector dimension */
    NumericTableIface::StorageLayout rLayout = r[0]->getDataLayout();

    algorithmFPType *x;      /* Input data           */
    algorithmFPType *d;      /* Resulting distances  */
    algorithmFPType *xxt;    /* Buffer for algorithmFPTypeediate results */

    if(isFull<algorithmFPType, cpu>(rLayout))
    {
        cosDistanceFull<algorithmFPType, cpu>( xTable, rTable );
    }
    else
    {
        if(isLower<algorithmFPType, cpu>(rLayout))
        {
            xxt = (algorithmFPType *)daal::services::daal_malloc(n * n * sizeof(algorithmFPType));
            if (!xxt) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

            BlockMicroTable<algorithmFPType, readOnly, cpu> *aMicroTable  =
                new BlockMicroTable<algorithmFPType, readOnly, cpu>(xTable);
            aMicroTable->getBlockOfRows(0, n, &x);

            char uplo, trans;
            algorithmFPType alpha, beta;
            /* Calculate X*X' */
            uplo  = 'U';
            trans = 'T';
            alpha = 1.0;
            beta  = 0.0;
            Blas<algorithmFPType, cpu>::xsyrk(&uplo, &trans, &n, &p, &alpha, x, &p, &beta, xxt, &n);

            aMicroTable->release();
            delete aMicroTable;

            algorithmFPType one = (algorithmFPType)1.0;
            for (size_t i = 0; i < n; i++)
            {
                if (xxt[i * n + i] != 0.0)
                {
                    xxt[i * n + i] = one / daal::internal::Math<algorithmFPType,cpu>::sSqrt(xxt[i * n + i]);
                }
            }

            PackedArrayMicroTable<algorithmFPType, writeOnly, cpu> *rPackedMicroTable =
                new PackedArrayMicroTable<algorithmFPType, writeOnly, cpu>
            (r[0]);
            rPackedMicroTable->getPackedArray(&d);

            /* Pack the results into output array */
            size_t dIndex = 0;
            for (size_t i = 0; i < n; i++)
            {
                for (size_t j = 0; j < i; j++)
                {
                    d[dIndex++] = one - xxt[i * n + j] * xxt[i * n + i] * xxt[j * n + j];
                }
                d[dIndex++] = one;
            }
            daal::services::daal_free(xxt);
            rPackedMicroTable->release();
            delete rPackedMicroTable;
        }
        else if(isUpper<algorithmFPType, cpu>(rLayout))
        {
            cosDistanceUpperPacked<algorithmFPType, cpu>( xTable, rTable );
        }
        else
        {
            this->_errors->add(services::ErrorIncorrectTypeOfOutputNumericTable); return;
        }
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

} // namespace cosine_distance

} // namespace algorithms

} // namespace daal
