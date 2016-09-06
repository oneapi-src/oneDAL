/* file: stump_train_aux.i */
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
//  Implementation of Fast method for Decision Stump algorithm.
//--
*/

#ifndef __STUMP_TRAIN_AUX_I__
#define __STUMP_TRAIN_AUX_I__

#include "daal_defines.h"
#include "service_memory.h"
#include "service_micro_table.h"

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

/**
 *  \brief Perform stump regression for data set X on responses Y with weights W
 */
template <Method method, typename algorithmFPtype, CpuType cpu>
void StumpTrainKernel<method, algorithmFPtype, cpu>::compute(size_t n, const NumericTable *const *a, stump::Model *r,
                                                             const Parameter *par)
{
    const NumericTable *xTable = a[0];
    const NumericTable *yTable = a[1];
    const NumericTable *wTable = (n >= 3 ? a[2] : 0);

    size_t nFeatures = xTable->getNumberOfColumns();
    size_t nVectors  = xTable->getNumberOfRows();
    NumericTable *xTableNoConst = const_cast<NumericTable *>(xTable);
    NumericTable *yTableNoConst = const_cast<NumericTable *>(yTable);
    NumericTable *wTableNoConst = const_cast<NumericTable *>(wTable);

    r->setNFeatures(nFeatures);

    algorithmFPtype sumW, sumWY, sumWYY;

    BlockDescriptor<algorithmFPtype> yBlock;
    yTableNoConst->getBlockOfColumnValues(0, 0, nVectors, readOnly, yBlock);
    algorithmFPtype *y = yBlock.getBlockPtr();

    BlockDescriptor<algorithmFPtype> wBlock;
    algorithmFPtype *w;
    if (wTable)
    {
        /* Here if array of weight is provided */
        wTableNoConst->getBlockOfColumnValues(0, 0, nVectors, readOnly, wBlock);
        w = wBlock.getBlockPtr();
    }
    else
    {
        /* Here if array of weight is not provided */
        algorithmFPtype weight = 1.0 / (algorithmFPtype)nVectors;
        w = (algorithmFPtype *) daal::services::daal_malloc(nVectors * sizeof(algorithmFPtype) );
        for (size_t i = 0; i < nVectors; i++)
        {
            w[i] = weight;
        }
    }

    size_t          splitFeature;
    algorithmFPtype splitPoint;
    algorithmFPtype leftValue;
    algorithmFPtype rightValue;

    doStumpRegression(nVectors, nFeatures, xTableNoConst, w, y,
                      &splitFeature, &splitPoint, &leftValue, &rightValue);

    r->splitFeature = splitFeature;

    daal::internal::BlockMicroTable<algorithmFPtype, writeOnly, cpu> mtValues(r->values.get());
    algorithmFPtype *stumpValues;
    mtValues.getBlockOfRows(0, 1, &stumpValues);
    stumpValues[0] = splitPoint;
    stumpValues[1] = leftValue;
    stumpValues[2] = rightValue;
    mtValues.release();

    yTableNoConst->releaseBlockOfColumnValues(yBlock);

    if (wTable)
    {
        /* Here if array of weight is provided */
        wTableNoConst->releaseBlockOfColumnValues(wBlock);
    }
    else
    {
        /* Here if array of weight is not provided */
        daal::services::daal_free(w);
    }

    return;
}

} // namespace daal::algorithms::stump::training::internal
}
}
}
} // namespace daal

#endif
