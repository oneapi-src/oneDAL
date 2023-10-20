/* file: gbt_regression_model_impl.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of the class defining the gradient boosted trees model
//--
*/

#ifndef __GBT_REGRESSION_MODEL_IMPL__
#define __GBT_REGRESSION_MODEL_IMPL__

#include "src/algorithms/dtrees/gbt/gbt_model_impl.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_model.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_model_builder.h"
#include "src/algorithms/regression/regression_model_impl.h"

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
    friend class gbt::regression::ModelBuilder;
    typedef gbt::internal::ModelImpl ImplType;
    typedef algorithms::regression::internal::ModelInternal RegressionImplType;

    ModelImpl(size_t nFeatures = 0) : RegressionImplType(nFeatures) {}
    ~ModelImpl() DAAL_C11_OVERRIDE {}

    virtual size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return RegressionImplType::getNumberOfFeatures(); }

    //Implementation of regression::Model
    virtual size_t numberOfTrees() const DAAL_C11_OVERRIDE;
    virtual void traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const DAAL_C11_OVERRIDE;
    virtual void traverseBF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const DAAL_C11_OVERRIDE;
    virtual void clear() DAAL_C11_OVERRIDE { ImplType::clear(); }
    virtual void traverseDFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const DAAL_C11_OVERRIDE;
    virtual void traverseBFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const DAAL_C11_OVERRIDE;

    virtual void setPredictionBias(double value) DAAL_C11_OVERRIDE;
    virtual double getPredictionBias() const DAAL_C11_OVERRIDE;

    virtual services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE;
    virtual services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE;

    virtual size_t getNumberOfTrees() const DAAL_C11_OVERRIDE;

private:
    /* global bias applied to predictions*/
    double _predictionBias;
};

} // namespace internal
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
