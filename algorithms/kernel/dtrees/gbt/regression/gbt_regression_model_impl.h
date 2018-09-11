/* file: gbt_regression_model_impl.h */
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
//  Implementation of the class defining the gradient boosted trees model
//--
*/

#ifndef __GBT_REGRESSION_MODEL_IMPL__
#define __GBT_REGRESSION_MODEL_IMPL__

#include "gbt_model_impl.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_model.h"
#include "../regression/regression_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace internal
{

class ModelImpl : public daal::algorithms::gbt::regression::Model,
    public algorithms::regression::internal::ModelInternal,
    public daal::algorithms::gbt::internal::ModelImpl
{
public:
    typedef gbt::internal::ModelImpl ImplType;
    typedef algorithms::regression::internal::ModelInternal RegressionImplType;

    ModelImpl(size_t nFeatures = 0) : RegressionImplType(nFeatures){}
    ~ModelImpl(){}

    virtual size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE{ return RegressionImplType::getNumberOfFeatures(); }

    //Implementation of regression::Model
    virtual size_t numberOfTrees() const DAAL_C11_OVERRIDE;
    virtual void traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor& visitor) const DAAL_C11_OVERRIDE;
    virtual void traverseBF(size_t iTree, algorithms::regression::TreeNodeVisitor& visitor) const DAAL_C11_OVERRIDE;
    virtual void clear() DAAL_C11_OVERRIDE { ImplType::clear(); }
    virtual void traverseDFS(size_t iTree, tree_utils::regression::TreeNodeVisitor& visitor) const DAAL_C11_OVERRIDE;
    virtual void traverseBFS(size_t iTree, tree_utils::regression::TreeNodeVisitor& visitor) const DAAL_C11_OVERRIDE;

    virtual services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE;
    virtual services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE;
};

} // namespace internal
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
