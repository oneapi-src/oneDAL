/* file: gbt_regression_model.h */
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
//  Implementation of the class defining the gradient boosted trees regression model
//--
*/

#ifndef __GBT_REGRESSION_MODEL_H__
#define __GBT_REGRESSION_MODEL_H__

#include "algorithms/regression/regression_model.h"
#include "algorithms/regression/tree_traverse.h"
#include "algorithms/tree_utils/tree_utils_regression.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
/**
 * @defgroup gbt_regression Gradient Boosted Trees Regression
 * \copydoc daal::algorithms::gbt::regression
 * @ingroup regression
 */
/**
 * \brief Contains classes for gradient boosted trees regression algorithm
 */
namespace regression
{
/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * @ingroup gbt_regression
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT_REGRESSION_MODEL"></a>
 * \brief %Base class for models trained with the gradient boosted trees regression algorithm
 *
 * \tparam modelFPType  Data type to store model data, double or float
 *
 * \par References
 *      - \ref regression::training::interface1::Batch "regression::training::Batch" class
 *      - \ref regression::prediction::interface1::Batch "regression::prediction::Batch" class
 */
class DAAL_EXPORT Model : public algorithms::regression::Model
{
public:
    DECLARE_MODEL(Model, algorithms::regression::Model);

    /**
     * Constructs the model
     * \param[in]  nFeatures Number of features in the dataset
     * \param[out] stat      Status of the model construction
     */
    static services::SharedPtr<Model> create(size_t nFeatures, services::Status *stat = NULL);

    /**
     *  Gets number of trees in the model
     *  \return number of trees
     */
    virtual size_t numberOfTrees() const = 0;

    /**
     *  Performs Depth First Traversal of i-th tree
     *  \param[in] iTree    Index of the tree to traverse
     *  \param[in] visitor  This object gets notified when tree nodes are visited
     *  \DAAL_DEPRECATED_USE{ Model::traverseDFS }
     */
    virtual void traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor& visitor) const = 0;

    /**
     *  Performs Breadth First Traversal of i-th tree
     *  \param[in] iTree    Index of the tree to traverse
     *  \param[in] visitor  This object gets notified when tree nodes are visited
     *  \DAAL_DEPRECATED_USE{ Model::traverseBFS }
     */
    virtual void traverseBF(size_t iTree, algorithms::regression::TreeNodeVisitor& visitor) const = 0;

    /**
     *  Removes all trees from the model
     */
    virtual void clear() = 0;

    /**
    *  Perform Depth First Traversal of i-th tree
    *  \param[in] iTree    Index of the tree to traverse
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    */
    virtual void traverseDFS(size_t iTree, tree_utils::regression::TreeNodeVisitor& visitor) const = 0;

    /**
    *  Perform Breadth First Traversal of i-th tree
    *  \param[in] iTree    Index of the tree to traverse
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    */
    virtual void traverseBFS(size_t iTree, tree_utils::regression::TreeNodeVisitor& visitor) const = 0;

protected:
    Model();
};
typedef services::SharedPtr<Model> ModelPtr;
typedef services::SharedPtr<const Model> ModelConstPtr;

/** @} */
} // namespace interface1
using interface1::Model;
using interface1::ModelPtr;
using interface1::ModelConstPtr;

}
}
}
}
#endif
