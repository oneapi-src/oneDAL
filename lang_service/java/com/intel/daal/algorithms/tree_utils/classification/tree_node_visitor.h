/* file: tree_node_visitor.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Callback class implementing TreeNodeVisitor interface
//--
*/
#ifndef __CLASSIFICATION_TREE_UTILS_NODE_VISITOR_H__
#define __CLASSIFICATION_TREE_UTILS_NODE_VISITOR_H__

#include <jni.h>
#include "daal.h"
#include "algorithms/tree_utils/tree_utils_classification.h"
#include "java_callback.h"

namespace daal
{
namespace classification
{
namespace tree_utils
{
/*
* \brief Callback class implementing TreeNodeVisitor interface
*/
struct JavaTreeNodeVisitor : public daal::algorithms::tree_utils::classification::TreeNodeVisitor, public daal::services::JavaCallback
{
    JavaTreeNodeVisitor(JavaVM *_jvm, jobject _javaObject) : daal::services::JavaCallback(_jvm, _javaObject) {}

    virtual ~JavaTreeNodeVisitor()
    {}

    virtual bool onLeafNode(const daal::algorithms::tree_utils::classification::LeafNodeDescriptor &desc) DAAL_C11_OVERRIDE;
    virtual bool onSplitNode(const daal::algorithms::tree_utils::SplitNodeDescriptor &desc) DAAL_C11_OVERRIDE;
};

}
}
}
#endif
