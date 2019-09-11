/* file: implicit_als_train_utils.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

#ifndef __IMPLICIT_ALS_TRAIN_UTILS_I__
#define __IMPLICIT_ALS_TRAIN_UTILS_I__

#include "implicit_als_train_utils.h"
#include "service_memory.h"
#include "service_sort.h"
#include "service_numeric_table.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
services::Status csr2csc(size_t nItems, size_t nUsers,
            const algorithmFPType *csrdata, const size_t *colIndices, const size_t *rowOffsets,
            algorithmFPType *cscdata, size_t *rowIndices, size_t *colOffsets)
{
    /* Convert CSR to COO */
    size_t dataSize = rowOffsets[nUsers] - rowOffsets[0];
    TArray<size_t, cpu> cooColIndicesPtr(dataSize);
    size_t *cooColIndices = cooColIndicesPtr.get();
    DAAL_CHECK_MALLOC(cooColIndices);

    daal_memcpy_s(cscdata, dataSize * sizeof(algorithmFPType), csrdata, dataSize * sizeof(algorithmFPType));
    daal_memcpy_s(cooColIndices, dataSize * sizeof(size_t), colIndices, dataSize * sizeof(size_t));

    /* Create array of row indices for COO data */
    for (size_t i = 1; i <= nUsers; i++)
    {
        size_t rowStart = rowOffsets[i-1] - 1;
        size_t rowEnd   = rowOffsets[i] - 1;
        for (size_t k = rowStart; k < rowEnd; k++)
        {
            rowIndices[k] = i;
        }
    }

    /* Sort arrays that represent data in COO format (values, column indices and row indices) over the column indices,
       and re-order arrays of values and row indices accordingly */
    daal::algorithms::internal::qSort<size_t, algorithmFPType, size_t, cpu>(dataSize, cooColIndices, cscdata, rowIndices);

    /* Create an array of columns offsets for the data in CSC format */
    size_t colOffset = 1;
    size_t colOffsetIndex = 0;
    for (; colOffsetIndex < cooColIndices[0]; colOffsetIndex++)
    {
        colOffsets[colOffsetIndex] = 1;
    }
    for (size_t i = 1; i < dataSize; i++)
    {
        if (cooColIndices[i] != cooColIndices[i - 1])
        {
            if (cooColIndices[i] == cooColIndices[i - 1] + 1)
            {
                colOffsets[colOffsetIndex++] = i + 1;
            }
            else
            {
                for (size_t k = cooColIndices[i - 1]; k < cooColIndices[i]; k++)
                {
                    colOffsets[colOffsetIndex++] = i + 1;
                }
            }
        }
    }
    for (size_t i = colOffsetIndex; i <= nItems; i++)
    {
        colOffsets[i] = rowOffsets[nUsers];
    }

    return Status();
}

}
}
}
}
}
#endif
