/* file: outlierdetection_multivariate_dense_default_kernel.h */
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
//  Declaration of template structs for multivariate outlier detection
//--
*/

#ifndef __MULTIVARIATE_OUTLIER_DETECTION_DENSE_DEFAULT_KERNEL_H__
#define __MULTIVARIATE_OUTLIER_DETECTION_DENSE_DEFAULT_KERNEL_H__

#include "numeric_table.h"
#include "outlier_detection_multivariate_types.h"

#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_math.h"

#include "outlierdetection_multivariate_kernel.h"

using namespace daal::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace multivariate_outlier_detection
{
namespace internal
{

template <typename AlgorithmFPType, CpuType cpu>
struct OutlierDetectionKernel<AlgorithmFPType, defaultDense, cpu> : public Kernel
{
    static const size_t blockSize = 1000;

    /** \brief Calculate Mahalanobis distance for a block of observations */
    inline void mahalanobisDistance(size_t nFeatures, size_t nVectors, AlgorithmFPType *data,
                                    AlgorithmFPType *location, AlgorithmFPType *invScatter, AlgorithmFPType *distance,
                                    AlgorithmFPType *buffer);

    /** \brief Detect outliers in the data from input micro-table
               and store resulting weights into output micro-table */
    inline void computeInternal(size_t nFeatures, size_t nVectors,
                                BlockMicroTable  <AlgorithmFPType, readOnly,  cpu> &mtA,
                                FeatureMicroTable<AlgorithmFPType, writeOnly, cpu> &mtR,
                                AlgorithmFPType *location, AlgorithmFPType *scatter, AlgorithmFPType threshold,
                                AlgorithmFPType *buffer);

    void compute(const NumericTable *a, NumericTable *r, const daal::algorithms::Parameter *par);
};

} // namespace internal

} // namespace multivariate_outlier_detection

} // namespace algorithms

} // namespace daal

#endif
