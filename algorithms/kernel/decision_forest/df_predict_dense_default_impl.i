/* file: df_predict_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of auxiliary functions for decision forest predict algorithms
//  (defaultDense) method.
//--
*/

#ifndef __DF_PREDICT_DENSE_DEFAULT_IMPL_I__
#define __DF_PREDICT_DENSE_DEFAULT_IMPL_I__

#include "df_model_impl.h"
#include "service_data_utils.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace prediction
{
namespace internal
{

//////////////////////////////////////////////////////////////////////////////////////////
// Common service function
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename TreeType, CpuType cpu>
const typename TreeType::NodeType::Base* findNode(const decision_forest::internal::Tree& t, const algorithmFPType* x)
{
    const TreeType& tree = static_cast<const TreeType&>(t);
    const typename TreeType::NodeType::Base* pNode = tree.top();
    if(tree.hasUnorderedFeatureSplits())
    {
        for(; pNode && pNode->isSplit();)
        {
            auto pSplit = TreeType::NodeType::castSplit(pNode);
            const int sn = (pSplit->featureUnordered ? (int(x[pSplit->featureIdx]) != int(pSplit->featureValue)) :
                daal::data_feature_utils::internal::SignBit<algorithmFPType, cpu>::get(pSplit->featureValue - x[pSplit->featureIdx]));
            pNode = pSplit->kid[sn];
        }
    }
    else
    {
        for(; pNode && pNode->isSplit();)
        {
            auto pSplit = TreeType::NodeType::castSplit(pNode);
            const int sn = daal::data_feature_utils::internal::SignBit<algorithmFPType, cpu>::get(pSplit->featureValue - x[pSplit->featureIdx]);
            pNode = pSplit->kid[sn];
        }
    }
    return pNode;
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif
