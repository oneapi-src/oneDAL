/* file: outlierdetection_multivariate_kernel.h */
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
//  Declaration of template structs for multivariate outlier detection
//--
*/

#ifndef __OUTLIERDETECTION_MULTIVARIATE_KERNEL_H__
#define __OUTLIERDETECTION_MULTIVARIATE_KERNEL_H__

#include "outlier_detection_multivariate.h"
#include "kernel.h"
#include "service_numeric_table.h"
#include "service_math.h"

using namespace daal::internal;
using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace multivariate_outlier_detection
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
struct OutlierDetectionKernel : public Kernel
{
    static const size_t blockSize = 1000;

    /** \brief Calculate Mahalanobis distance for a block of observations */
    inline void mahalanobisDistance(const size_t nFeatures, const size_t nVectors, const algorithmFPType * data, const algorithmFPType * location,
                                    const algorithmFPType * invScatter, algorithmFPType * distance, algorithmFPType * buffer);

    inline void defaultInitialization(algorithmFPType * location, algorithmFPType * scatter, algorithmFPType * threshold, const size_t nFeatures);

    /** \brief Detect outliers in the data from input micro-table
               and store resulting weights into output micro-table */
    inline Status computeInternal(const size_t nFeatures, const size_t nVectors, NumericTable & dataTable, NumericTable & resultTable,
                                  const algorithmFPType * location, const algorithmFPType * scatter, const algorithmFPType threshold,
                                  algorithmFPType * buffer);

    Status compute(NumericTable & dataTable, NumericTable * locationTable, NumericTable * scatterTable, NumericTable * thresholdTable,
                   NumericTable & resultTable);
};

/**
 * Added to support deprecated baconDense value
 */
template <typename algorithmFPType, CpuType cpu>
struct OutlierDetectionKernel<algorithmFPType, baconDense, cpu> : public Kernel
{
    Status compute(NumericTable & dataTable, NumericTable * locationTable, NumericTable * scatterTable, NumericTable * thresholdTable,
                   NumericTable & resultTable)
    {
        return services::Status();
    }
};

} // namespace internal

} // namespace multivariate_outlier_detection

} // namespace algorithms

} // namespace daal

#endif
