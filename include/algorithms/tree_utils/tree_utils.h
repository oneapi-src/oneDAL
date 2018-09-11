/* file: tree_utils.h */
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

#ifndef __TREE_UTILS__
#define __TREE_UTILS__

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

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__TREE_UTILS__NODEDESCRIPTOR"></a>
 * \brief %Struct containing base description of node in descision tree
 */
struct DAAL_EXPORT NodeDescriptor
{
    size_t level; /*!< Number of connections between the node and the root */
    double impurity; /*!< Measure of the homogeneity of the response variable at the node (i.e., the value of the criterion) */
    size_t nNodeSampleCount; /*!< Number of samples at the node */
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__TREE_UTILS__SPLITNODEDESCRIPTOR"></a>
 * \brief %Struct containing description of split node in descision tree
 */
struct DAAL_EXPORT SplitNodeDescriptor : public NodeDescriptor
{
    size_t featureIndex; /*!< Feature used for splitting the node */
    double featureValue; /*!< Threshold value at the node */
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__TREE_UTILS__TREENODEVISITOR"></a>
 * \brief %Interface of abstract visitor used in tree traversal methods.
 *
 * \par References
 *      - \ref decision_forest::regression::interface1::Model "Model" class
 *      - \ref decision_tree::regression::interface1::Model "Model" class
 */
template <typename LeafNodeDescriptorType>
class DAAL_EXPORT TreeNodeVisitor
{
public:
    /**
    *  This method is called by traversal method when a leaf node is visited.
    *  \param[in] desc     The structure containing description of the leaf visited
    *  \return This method should return false to cancel further search and true otherwise
    */
    virtual bool onSplitNode(const SplitNodeDescriptor &desc) = 0;

    /**
    *  This method is called by traversal method when a leaf node is visited.
    *  \param[in] desc     The structure containing description of the leaf visited
    *  \return This method should return false to cancel further search and true otherwise
    */
    virtual bool onLeafNode(const LeafNodeDescriptorType &desc) = 0;
};

} // interface1
using interface1::NodeDescriptor;
using interface1::SplitNodeDescriptor;
using interface1::TreeNodeVisitor;
} // tree_utils
} // algorithms
} // daal

#endif
