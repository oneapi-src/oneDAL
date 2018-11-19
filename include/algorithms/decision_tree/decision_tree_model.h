/* file: decision_tree_model.h */
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
//  Implementation of the class defining the Decision tree
//--
*/

#ifndef __DECISION_TREE_MODEL_H__
#define __DECISION_TREE_MODEL_H__

namespace daal
{
namespace algorithms
{

/**
 * @defgroup decision_tree Base Decision Tree
 * \brief Contains base classes for Decision tree algorithm
 * @ingroup training_and_prediction
 * @{
 */

/**
 * \brief Contains classes for Decision tree algorithm
 */
namespace decision_tree
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_TREE__PRUNING"></a>
 * \brief Pruning method for Decision tree algorithm
 */
enum Pruning
{
    none                = 0,    /*!< Do not prune */
    reducedErrorPruning = 1     /*!< Reduced error pruning */
};

} // namespace decision_tree

/** @} */
} // namespace algorithms
} // namespace daal

#endif
