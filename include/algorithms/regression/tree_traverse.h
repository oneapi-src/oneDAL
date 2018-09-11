/* file: tree_traverse.h */
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
//  Implementation of the class defining the Decision tree regression model
//--
*/

#ifndef __TREE_REGRESSION_TRAVERSE__
#define __TREE_REGRESSION_TRAVERSE__

namespace daal
{
namespace algorithms
{

/**
 * @defgroup trees_regression Tree regression
 * \copydoc daal::algorithms::trees::regression
 * @ingroup regression
 */

/**
 * \brief Contains classes for tree regression algorithms
 */
namespace regression
{

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup trees_regression
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__REGRESSION__TREENODEVISITOR"></a>
 * \brief %Interface of abstract visitor used in tree traversal methods. \DAAL_DEPRECATED
 *
 * \par References
 *      - \ref decision_forest::regression::interface1::Model "Model" class
 *      - \ref decision_tree::regression::interface1::Model "Model" class
 */
class DAAL_EXPORT TreeNodeVisitor
{
public:
    virtual ~TreeNodeVisitor(){}
    /**
    *  This method is called by traversal method when a leaf node is visited.
    *  \param[in] level    Level in the tree where this node is located
    *  \param[in] response The value of response given by that node
    *  \return This method should return false to cancel further search and true otherwise
    */
    virtual bool onLeafNode(size_t level, double response) = 0;
    /**
    *  This method is called by traversal method when a split node is visited.
    *  \param[in] level        Index of the feature used in a split criteria
    *  \param[in] featureIndex Index of the feature used as a split criteria in this node
    *  \param[in] featureValue Feature value used as a split criteria in this node
    *  \return This method should return false to cancel further search and true otherwise
    */
    virtual bool onSplitNode(size_t level, size_t featureIndex, double featureValue) = 0;
};

} // namespace interface1

using interface1::TreeNodeVisitor;

/** @} */
} // namespace regression
} // namespace algorithms
} // namespace daal

#endif
