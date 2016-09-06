/* file: stump_predict_impl.i */
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

#ifndef __STUMP_PREDICT_IMPL_I__
#define __STUMP_PREDICT_IMPL_I__


#include "algorithm.h"
#include "numeric_table.h"
#include "threading.h"
#include "daal_defines.h"
#include "service_micro_table.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace prediction
{
namespace internal
{

template <Method method, typename algorithmFPtype, CpuType cpu>
void StumpPredictKernel<method, algorithmFPtype, cpu>::compute(const NumericTable *xTable,
                                                               const stump::Model *m, size_t nr, NumericTable *rTables[],
                                                               const Parameter *par)
{
    size_t nVectors  = xTable->getNumberOfRows();
    NumericTable *xTableNoConst = const_cast<NumericTable *>(xTable);
    BlockDescriptor<algorithmFPtype> xBlock, rBlock;
    algorithmFPtype *x, *r;

    size_t splitFeature;
    algorithmFPtype splitPoint, leftValue, rightValue;

    stump::Model *model = const_cast<stump::Model *>(m);
    splitFeature = model->splitFeature;
    daal::internal::BlockMicroTable<algorithmFPtype, readOnly, cpu> mtValues(model->values.get());
    algorithmFPtype *stumpValues;
    mtValues.getBlockOfRows(0, 1, &stumpValues);
    splitPoint = stumpValues[0];
    leftValue  = stumpValues[1];
    rightValue = stumpValues[2];
    mtValues.release();

    rTables[0]->getBlockOfColumnValues(0, 0, nVectors, writeOnly, rBlock);
    r = rBlock.getBlockPtr();

    xTableNoConst->getBlockOfColumnValues(splitFeature, 0, nVectors, readOnly, xBlock);
    x = xBlock.getBlockPtr();

    for (size_t i = 0; i < nVectors; i++)
    {
        r[i] = ((x[i] < splitPoint) ? leftValue : rightValue);
    }

    xTableNoConst->releaseBlockOfColumnValues(xBlock);
    if(!this->_errors->isEmpty()) { return; }

    rTables[0]->releaseBlockOfColumnValues(rBlock);
    if(!this->_errors->isEmpty()) { return; }
}

} // namespace daal::algorithms::stump::prediction::internal
}
}
}
} // namespace daal

#endif
