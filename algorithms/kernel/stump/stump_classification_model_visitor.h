/* file: stump_classification_model_visitor.h */
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
//  Implementation of the decision stump model constructor.
//--
*/

#ifndef __STUMP_CLASSIFICATION_MODEL_VISITOR_H__
#define __STUMP_CLASSIFICATION_MODEL_VISITOR_H__

#include "algorithms/classifier/tree_traverse.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace classification
{
class StumpNodeVisitor : public daal::algorithms::tree_utils::classification::TreeNodeVisitor
{
public:
    StumpNodeVisitor(size_t nClasses)
        : daal::algorithms::tree_utils::classification::TreeNodeVisitor(),
          leftIsSet(false),
          changeZeroToMinusOne(false),
          splitFeature(0),
          splitValue(0),
          isOneLeaf(true),
          leftValue(0),
          rightValue(0)
    {
        if (nClasses == 2)
        {
            changeZeroToMinusOne = true;
        }
    }

    virtual bool onLeafNode(const tree_utils::classification::LeafNodeDescriptor & desc)
    {
        if (!leftIsSet)
        {
            leftValue = desc.label;
            leftIsSet = true;
        }
        rightValue = desc.label;
        if (changeZeroToMinusOne)
        {
            if (rightValue == 0)
            {
                rightValue = -1;
            }
            if (leftValue == 0)
            {
                leftValue = -1;
            }
        }
        return true;
    }

    virtual bool onSplitNode(const tree_utils::classification::SplitNodeDescriptor & desc)
    {
        isOneLeaf    = false;
        splitValue   = desc.featureValue;
        splitFeature = desc.featureIndex;
        return true;
    }

    bool leftIsSet;
    bool isOneLeaf;
    bool changeZeroToMinusOne;
    size_t splitFeature;
    double leftValue;
    double rightValue;
    double splitValue;
};

} // namespace classification
} // namespace stump
} // namespace algorithms
} // namespace daal

#endif
