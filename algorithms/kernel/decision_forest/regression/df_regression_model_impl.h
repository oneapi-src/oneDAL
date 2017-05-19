/* file: df_regression_model_impl.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of the class defining the decision forest model
//--
*/

#ifndef __DF_REGRESSION_MODEL_IMPL__
#define __DF_REGRESSION_MODEL_IMPL__

#include "df_model_impl.h"
#include "algorithms/decision_forest/decision_forest_regression_model.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace internal
{

class ModelImpl : public daal::algorithms::decision_forest::regression::Model,
    public daal::algorithms::decision_forest::internal::ModelImpl
{
public:
    typedef decision_forest::internal::ModelImpl ImplType;
    typedef decision_forest::internal::TreeImpRegression<> TreeType;

    ModelImpl(size_t nFeatures = 0) : _nFeatures(nFeatures){}
    ~ModelImpl(){}

    virtual size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE{ return _nFeatures; }

    //Implementation of decision_forest::regression::Model
    virtual size_t numberOfTrees() const DAAL_C11_OVERRIDE;
    virtual void traverseDF(size_t iTree, decision_forest::regression::NodeVisitor& visitor) const DAAL_C11_OVERRIDE;
    virtual void traverseBF(size_t iTree, NodeVisitor& visitor) const DAAL_C11_OVERRIDE;

protected:
    size_t _nFeatures;
};

} // namespace internal
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
