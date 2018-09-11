/* file: tree_utils_classification.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
//  Implementation of the class defining the Decision tree classification model
//--
*/

#ifndef __TREE_UTILS_CLASSIFICATION__
#define __TREE_UTILS_CLASSIFICATION__

#include "tree_utils.h"

namespace daal
{
namespace algorithms
{

/**
 * @defgroup tree_utils Tree utils
 * \brief Contains classes for work with the tree-based algorithms
 * @ingroup training_and_prediction
 */
namespace tree_utils
{

namespace classification
{

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__TREE_UTILS__CLASSIFICATION__LEAFNODEDESCRIPTOR"></a>
 * \brief %Struct containing description of leaf node in classification descision tree
 */
struct DAAL_EXPORT LeafNodeDescriptor : public NodeDescriptor
{
    size_t label; /*!< Label to be predicted when reaching the leaf */
};

typedef daal::algorithms::tree_utils::TreeNodeVisitor<LeafNodeDescriptor> TreeNodeVisitor;
typedef daal::algorithms::tree_utils::SplitNodeDescriptor SplitNodeDescriptor;

} // interface1
using interface1::TreeNodeVisitor;
using interface1::SplitNodeDescriptor;
using interface1::LeafNodeDescriptor;
} // classification
} // tree_utils
} // algorithms
} // daal

#endif
