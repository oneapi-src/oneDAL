/* file: stump_train_aux.i */
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
//  Implementation of Fast method for Decision Stump algorithm.
//--
*/

#ifndef __STUMP_TRAIN_AUX_I__
#define __STUMP_TRAIN_AUX_I__

#include "daal_defines.h"
#include "service_memory.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace training
{
namespace internal
{
using namespace daal::internal;

/**
 *  \brief Perform stump regression for data set X on responses Y with weights W
 */
template <Method method, typename algorithmFPtype, CpuType cpu>
services::Status StumpTrainKernel<method, algorithmFPtype, cpu>::compute(size_t n, const NumericTable * const * a, stump::Model * r,
                                                                         const Parameter * par)
{
    const NumericTable * xTable = a[0];
    const NumericTable * yTable = a[1];
    const NumericTable * wTable = (n >= 3 ? a[2] : 0);

    const size_t nFeatures = xTable->getNumberOfColumns();
    const size_t nVectors  = xTable->getNumberOfRows();
    r->setNFeatures(nFeatures);

    services::Status s;
    ReadColumns<algorithmFPtype, cpu> wBlock(const_cast<NumericTable *>(wTable), 0, 0, nVectors);
    TArray<algorithmFPtype, cpu> wArray(wTable ? 0 : nVectors);
    if (wTable)
    {
        /* Here if array of weight is provided */
        DAAL_CHECK_STATUS(s, wBlock.status());
    }
    else
    {
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nVectors, sizeof(algorithmFPtype));
        DAAL_CHECK(wArray.get(), services::ErrorMemoryAllocationFailed);
        /* Here if array of weight is not provided */
        const algorithmFPtype weight = 1.0 / algorithmFPtype(nVectors);
        algorithmFPtype * w          = wArray.get();
        for (size_t i = 0; i < nVectors; i++) w[i] = weight;
    }

    algorithmFPtype splitPoint;
    algorithmFPtype leftValue;
    algorithmFPtype rightValue;
    size_t splitFeature;
    {
        ReadColumns<algorithmFPtype, cpu> y(const_cast<NumericTable *>(yTable), 0, 0, nVectors);
        DAAL_CHECK_STATUS(s, y.status());
        doStumpRegression(nVectors, nFeatures, xTable, (wTable ? wBlock.get() : wArray.get()), y.get(), splitFeature, splitPoint, leftValue,
                          rightValue);
    }

    r->setSplitFeature(splitFeature);
    r->setSplitValue<algorithmFPtype>(splitPoint);
    r->setLeftSubsetAverage<algorithmFPtype>(leftValue);
    r->setRightSubsetAverage<algorithmFPtype>(rightValue);

    return s;
}

} // namespace internal
} // namespace training
} // namespace stump
} // namespace algorithms
} // namespace daal

#endif
