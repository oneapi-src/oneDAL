/* file: outlierdetection_univariate_kernel.h */
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
//  Declaration of template structs for Outliers Detection.
//--
*/

#ifndef __UNIVAR_OUTLIERDETECTION_KERNEL_H__
#define __UNIVAR_OUTLIERDETECTION_KERNEL_H__

#include "outlier_detection_univariate.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_math.h"
#include "service_memory.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace univariate_outlier_detection
{
namespace internal
{

using namespace daal::internal;
using namespace daal::internal;
using namespace daal::services;

template <typename algorithmFPType, Method method, CpuType cpu>
struct OutlierDetectionKernel : public Kernel
{
    static const size_t blockSize = 1000;

    /** \brief Detect outliers in the data from input table
               and store resulting weights into output table */
    inline static Status computeInternal(size_t nFeatures, size_t nVectors,
                                       NumericTable &dataTable,
                                       NumericTable &resultTable,
                                       const algorithmFPType *location, const algorithmFPType *scatter, algorithmFPType *invScatter,
                                       const algorithmFPType *threshold)
    {
        ReadRows<algorithmFPType, cpu> dataBlock(dataTable);
        WriteOnlyRows<algorithmFPType, cpu> resultBlock(resultTable);

        const algorithmFPType zero(0.0);
        const algorithmFPType one(1.0);

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nFeatures; j++)
        {
            invScatter[j] = one;
            if (scatter[j] != zero)
            {
                invScatter[j] = one / scatter[j];
            }
        }

        size_t nBlocks = nVectors / blockSize;
        if (nBlocks * blockSize < nVectors)
        {
            nBlocks++;
        }

        /* Process input data table in blocks */
        for (size_t iBlock = 0; iBlock < nBlocks; iBlock++)
        {
            size_t startRow = iBlock * blockSize;
            size_t nRowsInBlock = blockSize;
            if (startRow + nRowsInBlock > nVectors)
            {
                nRowsInBlock = nVectors - startRow;
            }

            const algorithmFPType *data = dataBlock.next(startRow, nRowsInBlock);
            algorithmFPType *weight = resultBlock.next(startRow, nRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS(dataBlock);
            DAAL_CHECK_BLOCK_STATUS(resultBlock);

            const algorithmFPType *dataPtr = data;
            algorithmFPType *weightPtr = weight;
            algorithmFPType diff;
            for (size_t i = 0;  i < nRowsInBlock; i++, dataPtr += nFeatures, weightPtr += nFeatures)
            {
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < nFeatures; j++)
                {
                    weightPtr[j] = one;
                    diff = daal::internal::Math<algorithmFPType, cpu>::sFabs(dataPtr[j] - location[j]);
                    if (scatter[j] != zero)
                    {
                        /* Here if scatter is greater than zero */
                        if (diff * invScatter[j] > threshold[j]) { weightPtr[j] = zero; }
                    }
                    else
                    {
                        /* Here if scatter is equal to zero */
                        if (diff > zero) { weightPtr[j] = zero; }
                    }
                }
            }
        }
        return Status();
    }

    /** \brief Detect outliers in the data from input numeric table
               and store resulting weights into output numeric table */
    Status compute(NumericTable &dataTable, NumericTable &resultTable,
                   NumericTable *locationTable,
                   NumericTable *scatterTable,
                   NumericTable *thresholdTable);

    void defaultInitialization(algorithmFPType *location,
                               algorithmFPType *scatter,
                               algorithmFPType *threshold,
                               const size_t nFeatures);
};

} // namespace internal

} // namespace univariate_outlier_detection

} // namespace algorithms

} // namespace daal

#endif
