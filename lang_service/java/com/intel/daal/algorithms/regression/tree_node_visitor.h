/* file: tree_node_visitor.h */
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
//  Callback class implementing TreeNodeVisitor interface
//--
*/
#ifndef __REGRESSION_TREE_NODE_VISITOR_H__
#define __REGRESSION_TREE_NODE_VISITOR_H__
#include <jni.h>
#include "daal.h"
#include "java_callback.h"
#include "algorithms/regression/tree_traverse.h"

namespace daal
{
namespace regression
{
/*
* \brief Callback class implementing TreeNodeVisitor interface
*/
struct JavaTreeNodeVisitor : public daal::algorithms::regression::TreeNodeVisitor, public daal::services::JavaCallback
{
    JavaTreeNodeVisitor(JavaVM * _jvm, jobject _javaObject) : daal::services::JavaCallback(_jvm, _javaObject) {}

    virtual ~JavaTreeNodeVisitor() {}

    virtual bool onLeafNode(size_t level, double response) DAAL_C11_OVERRIDE;
    virtual bool onSplitNode(size_t level, size_t featureIndex, double featureValue) DAAL_C11_OVERRIDE;
};
} // namespace regression
} // namespace daal
#endif
