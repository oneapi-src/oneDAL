/* file: outlierdetection_univariate_dense_default_kernel.h */
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
//  Declaration of template structs for univariate outlier detection
//--
*/

#ifndef __UNIVAR_OUTLIERDETECTION_DENSE_DEFAULT_KERNEL_H__
#define __UNIVAR_OUTLIERDETECTION_DENSE_DEFAULT_KERNEL_H__

#include "numeric_table.h"
#include "outlier_detection_univariate_types.h"

#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_memory.h"
#include "service_math.h"

#include "outlierdetection_univariate_kernel.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace univariate_outlier_detection
{
namespace internal
{

template <typename AlgorithmFPType, CpuType cpu>
struct OutlierDetectionKernel<AlgorithmFPType, defaultDense, cpu> : public Kernel
{
    static const size_t blockSize = 1000;

    /** \brief Detect outliers in the data from input micro-table
               and store resulting weights into output micro-table */
    inline static void computeInternal(size_t nFeatures, size_t nVectors,
                                       BlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA,
                                       BlockMicroTable<AlgorithmFPType, writeOnly, cpu> &mtR,
                                       AlgorithmFPType *location, AlgorithmFPType *scatter, AlgorithmFPType *invScatter,
                                       AlgorithmFPType *threshold)
    {
        AlgorithmFPType zero = (AlgorithmFPType)0.0;
        AlgorithmFPType one = (AlgorithmFPType)1.0;

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
            AlgorithmFPType *data, *weight;
            mtA.getBlockOfRows(startRow, nRowsInBlock, &data);
            mtR.getBlockOfRows(startRow, nRowsInBlock, &weight);

            AlgorithmFPType *dataPtr = data;
            AlgorithmFPType *weightPtr = weight;
            AlgorithmFPType diff;
            for (size_t i = 0;  i < nRowsInBlock; i++, dataPtr += nFeatures, weightPtr += nFeatures)
            {
              PRAGMA_IVDEP
              PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < nFeatures; j++)
                {
                    weightPtr[j] = one;
                    diff = daal::internal::Math<AlgorithmFPType,cpu>::sFabs(dataPtr[j] - location[j]);
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

            mtA.release();
            mtR.release();
        }
    }

    /** \brief Detect outliers in the data from input numeric table
               and store resulting weights into output numeric table */
    void compute(const NumericTable *a, NumericTable *r, const daal::algorithms::Parameter *par);
};

} // namespace internal

} // namespace univariate_outlier_detection

} // namespace algorithms

} // namespace daal

#endif
