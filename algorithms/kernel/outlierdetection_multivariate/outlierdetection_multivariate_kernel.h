/* file: outlierdetection_multivariate_kernel.h */
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
//  Declaration of template structs for Outliers Detection.
//--
*/

#ifndef __MULTIVARIATE_OUTLIER_DETECTION_KERNEL_H__
#define __MULTIVARIATE_OUTLIER_DETECTION_KERNEL_H__

#include "numeric_table.h"
#include "outlier_detection_multivariate.h"
#include "outlier_detection_multivariate_types.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace multivariate_outlier_detection
{
namespace internal
{

template<CpuType cpu>
struct TemporaryInitialization : public multivariate_outlier_detection::InitIface
{
    size_t nFeatures;

    explicit TemporaryInitialization(size_t nFeatures) : nFeatures(nFeatures) {}

    virtual void operator()(NumericTable *data,
                            NumericTable *location,
                            NumericTable *scatter,
                            NumericTable *threshold)
    {
        BlockDescriptor<double> locationBlock;
        location->getBlockOfRows(0, 1, writeOnly, locationBlock);
        double *locationArray = locationBlock.getBlockPtr();
        BlockDescriptor<double> scatterBlock;
        scatter->getBlockOfRows(0, nFeatures, writeOnly, scatterBlock);
        double *scatterArray = scatterBlock.getBlockPtr();
        BlockDescriptor<double> thresholdBlock;
        threshold->getBlockOfRows(0, 1, writeOnly, thresholdBlock);
        double *thresholdArray = thresholdBlock.getBlockPtr();

        for(size_t i = 0; i < nFeatures; i++)
        {
            locationArray[i]  = 0.0;
            for(size_t j = 0; j < nFeatures; j++)
            {
                scatterArray[i * nFeatures + j] = 0.0;
            }
            scatterArray[i * nFeatures + i]   = 1.0;
        }
        thresholdArray[0] = 3.0;

        location->releaseBlockOfRows(locationBlock);
        scatter->releaseBlockOfRows(scatterBlock);
        threshold->releaseBlockOfRows(thresholdBlock);
    }
};

template <typename AlgorithmFPType, Method method, CpuType cpu>
struct OutlierDetectionKernel : public Kernel
{
    void compute(const NumericTable *a, NumericTable *r, const daal::algorithms::Parameter *par);
};

} // namespace internal

} // namespace multivariate_outlier_detection

} // namespace algorithms

} // namespace daal

#endif
