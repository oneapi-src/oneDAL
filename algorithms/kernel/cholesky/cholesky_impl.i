/* file: cholesky_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of cholesky algorithm
//--
*/

#include "service_numeric_table.h"
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
    const DAAL_INT dim = (DAAL_INT)(aTable->getNumberOfColumns());   /* Dimension of input feature vectors */

    const NumericTableIface::StorageLayout iLayout = aTable->getDataLayout();
    const NumericTableIface::StorageLayout rLayout = r->getDataLayout();

    WriteOnlyRows<algorithmFPType, cpu> rowsR;
    WriteOnlyPacked<algorithmFPType, cpu> packedR;

    algorithmFPType *L = nullptr;
    if(isFull<algorithmFPType, cpu>(rLayout))
    {
        rowsR.set(*r, 0, dim);
        DAAL_CHECK_BLOCK_STATUS(rowsR);
        L = rowsR.get();
    }
    else
    {
        packedR.set(r);
        DAAL_CHECK_BLOCK_STATUS(packedR);
        L = packedR.get();
    }

    Status s;
    if(isFull<algorithmFPType, cpu>(iLayout))
    {
        ReadRows<algorithmFPType, cpu> rowsA(*aTable, 0, dim);
        DAAL_CHECK_BLOCK_STATUS(rowsA);
        s = copyMatrix(iLayout, rowsA.get(), rLayout, L, dim);
    }
    else
    {
        ReadPacked<algorithmFPType, cpu> packedA(*aTable);
        DAAL_CHECK_BLOCK_STATUS(packedA);
        s = copyMatrix(iLayout, packedA.get(), rLayout, L, dim);
    }
    return s.ok() ? performCholesky(rLayout, L, dim) : s;
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
