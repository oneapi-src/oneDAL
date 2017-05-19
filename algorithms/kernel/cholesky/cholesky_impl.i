/* file: cholesky_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of cholesky algorithm
//--
*/

#include "service_micro_table.h"
#include "service_lapack.h"

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace cholesky
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
bool isFull(NumericTableIface::StorageLayout rLayout);

/**
 *  \brief Kernel for Cholesky calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
Status CholeskyKernel<algorithmFPType, method, cpu>::compute(NumericTable *aTable, NumericTable *r, const daal::algorithms::Parameter *par)
{
    DAAL_INT dim = (DAAL_INT)(aTable->getNumberOfColumns());   /* Dimension of input feature vectors */

    algorithmFPType *A;
    algorithmFPType *L;

    NumericTableIface::StorageLayout iLayout = aTable->getDataLayout();
    NumericTableIface::StorageLayout rLayout = r->getDataLayout();

    BlockMicroTable<algorithmFPType, readOnly, cpu> *aMicroTable = NULL;
    BlockMicroTable<algorithmFPType, writeOnly, cpu> *rMicroTable = NULL;
    PackedArrayMicroTable<algorithmFPType, readOnly, cpu> *aPackedMicroTable = NULL;
    PackedArrayMicroTable<algorithmFPType, writeOnly, cpu> *rPackedMicroTable = NULL;

    if (isFull<algorithmFPType, cpu>(iLayout))
    {
        aMicroTable = new BlockMicroTable<algorithmFPType, readOnly, cpu>(aTable);
        aMicroTable->getBlockOfRows(0, dim, &A);
    }
    else
    {
        aPackedMicroTable = new PackedArrayMicroTable<algorithmFPType, readOnly, cpu>(aTable);
        aPackedMicroTable->getPackedArray(&A);
    }

    if (isFull<algorithmFPType, cpu>(rLayout))
    {
        rMicroTable = new BlockMicroTable<algorithmFPType, writeOnly, cpu>(r);
        rMicroTable->getBlockOfRows(0, dim, &L);
    }
    else
    {
        rPackedMicroTable = new PackedArrayMicroTable<algorithmFPType, writeOnly, cpu>(r);
        rPackedMicroTable->getPackedArray(&L);
    }

    Status s = copyMatrix(iLayout, A, rLayout, L, dim);
    (isFull<algorithmFPType, cpu>(iLayout)) ? aMicroTable->release() : aPackedMicroTable->release();

    if(s)
    {
        s = performCholesky(rLayout, L, dim);
        (isFull<algorithmFPType, cpu>(rLayout)) ? rMicroTable->release() : rPackedMicroTable->release();
    }

    delete aMicroTable;
    delete rMicroTable;
    delete aPackedMicroTable;
    delete rPackedMicroTable;

    return s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status CholeskyKernel<algorithmFPType, method, cpu>::copyMatrix(NumericTableIface::StorageLayout iLayout,
    const algorithmFPType *pA, NumericTableIface::StorageLayout rLayout, algorithmFPType *pL, DAAL_INT dim) const
{
    if(isFull<algorithmFPType, cpu>(rLayout))
    {
        if(!copyToFullMatrix(iLayout, pA, pL, dim))
            return Status(ErrorIncorrectTypeOfInputNumericTable);

    }
    else
    {
        if(!copyToLowerTrianglePacked(iLayout, pA, pL, dim))
            return Status(ErrorIncorrectTypeOfOutputNumericTable);

    }
    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status CholeskyKernel<algorithmFPType, method, cpu>::performCholesky(NumericTableIface::StorageLayout rLayout,
                                                                   algorithmFPType *pL, DAAL_INT dim)
{
    DAAL_INT info;
    char uplo = 'U';

    if (isFull<algorithmFPType, cpu>(rLayout))
    {
        Lapack<algorithmFPType, cpu>::xpotrf(&uplo, &dim, pL, &dim, &info);
    }
    else if (rLayout == NumericTableIface::lowerPackedTriangularMatrix)
    {
        Lapack<algorithmFPType, cpu>::xpptrf(&uplo, &dim, pL, &info);
    }
    else
    {
        return Status(ErrorIncorrectTypeOfOutputNumericTable);
    }

    if(info > 0)
        return Status(Error::create(services::ErrorInputMatrixHasNonPositiveMinor, services::Minor, (int)info));

    return info < 0 ? Status(services::ErrorCholeskyInternal) : Status();
}

template <typename algorithmFPType, CpuType cpu>
bool isFull(NumericTableIface::StorageLayout layout)
{
    int layoutInt = (int) layout;
    if (packed_mask & layoutInt && NumericTableIface::csrArray != layoutInt)
    {
        return false;
    }
    return true;
}

template <typename algorithmFPType, Method method, CpuType cpu>
bool CholeskyKernel<algorithmFPType, method, cpu>::copyToFullMatrix(NumericTableIface::StorageLayout iLayout,
    const algorithmFPType *pA, algorithmFPType *pL, DAAL_INT dim) const
{
    if (isFull<algorithmFPType, cpu>(iLayout))
    {
        for (DAAL_INT i = 0; i < dim; i++)
        {
            for (DAAL_INT j = 0; j <= i; j++)
            {
                pL[i * dim + j] = pA[i * dim + j];
            }
            for (DAAL_INT j = (i + 1); j < dim; j++)
            {
                pL[i * dim + j] = 0;
            }
        }
    }
    else if (iLayout == NumericTableIface::lowerPackedSymmetricMatrix)
    {
        DAAL_INT ind = 0;
        for (DAAL_INT i = 0; i < dim; i++)
        {
            for (DAAL_INT j = 0; j <= i; j++)
            {
                pL[i * dim + j] = pA[ind++];
            }
            for (DAAL_INT j = (i + 1); j < dim; j++)
            {
                pL[i * dim + j] = 0;
            }
        }
    }
    else if (iLayout == NumericTableIface::upperPackedSymmetricMatrix)
    {
        DAAL_INT ind = 0;
        for (DAAL_INT j = 0; j < dim; j++)
        {
            for (DAAL_INT i = 0; i < j ; i++)
            {
                pL[i * dim + j] = 0;
            }
            for (DAAL_INT i = j; i < dim; i++)
            {
                pL[i * dim + j] = pA[ind++];
            }
        }
    }
    else
    {
        return false;
    }
    return true;
}

template <typename algorithmFPType, Method method, CpuType cpu>
bool CholeskyKernel<algorithmFPType, method, cpu>::copyToLowerTrianglePacked(NumericTableIface::StorageLayout iLayout,
    const algorithmFPType *pA, algorithmFPType *pL, DAAL_INT dim) const
{
    if (isFull<algorithmFPType, cpu>(iLayout))
    {
        DAAL_INT ind = 0;
        for (DAAL_INT i = 0; i < dim; i++)
        {
            for (DAAL_INT j = 0; j <= i; j++)
            {
                pL[ind++] = pA[i * dim + j];
            }
        }
    }
    else if (iLayout == NumericTableIface::lowerPackedSymmetricMatrix)
    {
        for (DAAL_INT i = 0; i < dim * (dim + 1) / 2; i++)
        {
            pL[i] = pA[i];
        }
    }
    else if (iLayout == NumericTableIface::upperPackedSymmetricMatrix)
    {
        DAAL_INT ind = 0;
        for(DAAL_INT j = 0; j < dim; j++)
        {
            for(DAAL_INT i = 0; i <= j ; i++)
            {
                pL[ind++] = pA[(dim * i - i * (i - 1) / 2 - i) + j];
            }
        }
    }
    else
    {
        return false;
    }
    return true;
}

} // namespace daal::internal
} // namespace daal::cholesky
}
} // namespace daal
