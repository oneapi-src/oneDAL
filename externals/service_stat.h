/* file: service_stat.h */
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
//  Template wrappers for STAT functions.
//--
*/


#ifndef __SERVICE_STAT_H__
#define __SERVICE_STAT_H__

#include "daal_defines.h"
#include "service_memory.h"

#include "service_stat_mkl.h"

namespace daal
{
namespace internal
{

/*
// Template functions definition
*/
template<typename fpType, CpuType cpu, template<typename, CpuType> class _impl=mkl::MklStatistics>
struct Statistics
{
    typedef typename _impl<fpType,cpu>::SizeType   SizeType;
    typedef typename _impl<fpType,cpu>::MethodType MethodType;
    typedef typename _impl<fpType,cpu>::ErrorType  ErrorType;

    static ErrorType xcp(fpType *data, SizeType nFeatures, SizeType nVectors, fpType *nPreviousObservations, fpType *sum,
            fpType *crossProduct, MethodType method)
    {
        return _impl<fpType,cpu>::xcp(data, nFeatures, nVectors, nPreviousObservations, sum, crossProduct, method);
    }

    static ErrorType xxcp_weight(fpType *data, SizeType nFeatures, SizeType nVectors, fpType *weight, fpType *accumWeight, fpType *mean,
                    fpType *crossProduct, MethodType method)
    {
        return _impl<fpType,cpu>::xxcp_weight(data, nFeatures, nVectors, weight, accumWeight, mean, crossProduct, method);
    }

    static ErrorType x2c_mom(fpType *data, SizeType nFeatures, SizeType nVectors, fpType *variance, MethodType method)
    {
        return _impl<fpType,cpu>::x2c_mom(data, nFeatures, nVectors, variance, method);
    }

    static ErrorType xoutlierdetection(fpType *data, SizeType nFeatures, SizeType nVectors, SizeType nParams,
                          fpType *baconParams, fpType *baconWeights)
    {
        return _impl<fpType,cpu>::xoutlierdetection(data, nFeatures, nVectors,  nParams, baconParams, baconWeights);
    }

    static ErrorType xLowOrderMoments(fpType *data, SizeType nFeatures, SizeType nVectors, MethodType method,
                         fpType *sum, fpType *mean, fpType *secondRawMoment,
                         fpType *variance, fpType *variation)
    {
        return _impl<fpType,cpu>::xLowOrderMoments(data, nFeatures, nVectors, method, sum, mean, secondRawMoment, variance, variation);
    }

    static ErrorType xSumAndVariance(fpType *data, SizeType nFeatures, SizeType nVectors, fpType *nPreviousObservations,
                        MethodType method, fpType *sum, fpType *mean, fpType *secondRawMoment,
                        fpType *variance)
    {
        return _impl<fpType,cpu>::xSumAndVariance(data, nFeatures, nVectors, nPreviousObservations, method, sum, mean, secondRawMoment, variance);
    }

    static ErrorType xQuantiles(fpType *data, SizeType nFeatures, SizeType nVectors, SizeType quantOrderN, fpType* quantOrder, fpType *quants)
    {
        return _impl<fpType,cpu>::xQuantiles(data, nFeatures, nVectors, quantOrderN, quantOrder, quants);
    }

    static ErrorType xSort(fpType *data, SizeType nFeatures, SizeType nVectors, fpType *sortedData)
    {
        return _impl<fpType,cpu>::xSort(data, nFeatures, nVectors, sortedData);
    }
};

} // namespace internal
} // namespace daal

#endif
