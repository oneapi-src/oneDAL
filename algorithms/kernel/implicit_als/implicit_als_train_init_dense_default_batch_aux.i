/* file: implicit_als_train_init_dense_default_batch_aux.i */
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
//  Implementation of auxiliary functions for impicit ALS initialization
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_INIT_DENSE_DEFAULT_BATCH_AUX_I__
#define __IMPLICIT_ALS_TRAIN_INIT_DENSE_DEFAULT_BATCH_AUX_I__

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
ImplicitALSInitTask<algorithmFPType, cpu>::ImplicitALSInitTask(NumericTable *itemsFactorsTable,
            ImplicitALSInitKernelBase<algorithmFPType, cpu> *algorithm) :
            mtItemsFactors(itemsFactorsTable),
            itemsFactors(NULL)
{
    size_t nItems = mtItemsFactors.getFullNumberOfRows();
    size_t nRowsRead = mtItemsFactors.getBlockOfRows(0, nItems, &itemsFactors);
    if (nRowsRead < nItems)
    {
        algorithm->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable);
        return;
    }
}

template <typename algorithmFPType, CpuType cpu>
ImplicitALSInitTask<algorithmFPType, cpu>::~ImplicitALSInitTask()
{
    mtItemsFactors.release();
}

}
}
}
}
}
}

#endif
