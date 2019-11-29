/* file: stump_regression_model_visitor.h */
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

#ifndef __STUMP_REGRESSION_MODEL_VISITOR_H__
#define __STUMP_REGRESSION_MODEL_VISITOR_H__

#include "algorithms/regression/tree_traverse.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace regression
{
/** Visitor class implementing TreeNodeVisitor interface, prints out tree nodes of the model when it is called back by model traversal method */
class StumpNodeVisitor : public daal::algorithms::tree_utils::regression::TreeNodeVisitor
{
public:
    StumpNodeVisitor() : leftIsSet(false), splitFeature(0), isOneLeaf(true), leftValue(0), rightValue(0), splitValue(0) {}
    virtual bool onLeafNode(const daal::algorithms::tree_utils::regression::LeafNodeDescriptor & desc)
    {
        if (!leftIsSet)
        {
            leftValue = desc.response;
            leftIsSet = true;
        }
        rightValue = desc.response;
        return true;
    }

    virtual bool onSplitNode(const daal::algorithms::tree_utils::regression::SplitNodeDescriptor & desc)
    {
        isOneLeaf    = false;
        splitFeature = desc.featureIndex;
        splitValue   = desc.featureValue;
        return true;
    }
    bool leftIsSet;
    bool isOneLeaf;
    size_t splitFeature;
    double leftValue;
    double rightValue;
    double splitValue;
};

} // namespace regression
} // namespace stump
} // namespace algorithms
} // namespace daal

#endif
