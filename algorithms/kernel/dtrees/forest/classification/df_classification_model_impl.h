/* file: df_classification_model_impl.h */
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
//  Implementation of the class defining the decision forest model
//--
*/

#ifndef __DTREES_CLASSIFICATION_MODEL_IMPL__
#define __DTREES_CLASSIFICATION_MODEL_IMPL__

#include "dtrees_model_impl.h"
#include "algorithms/decision_forest/decision_forest_classification_model.h"
#include "classifier_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace internal
{

class ModelImpl : public decision_forest::classification::Model, public algorithms::classifier::internal::ModelInternal, public dtrees::internal::ModelImpl
{
public:
    typedef dtrees::internal::ModelImpl ImplType;
    typedef algorithms::classifier::internal::ModelInternal ClassifierImplType;
    typedef dtrees::internal::TreeImpClassification<> TreeType;
    ModelImpl(size_t nFeatures = 0) : ClassifierImplType(nFeatures) {}
    ~ModelImpl(){}

    //Implementation of classifier::Model
    virtual size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return ClassifierImplType::getNumberOfFeatures(); }
    virtual size_t getNFeatures() const DAAL_C11_OVERRIDE { return getNumberOfFeatures(); }
    virtual void setNFeatures(size_t nFeatures) DAAL_C11_OVERRIDE { ClassifierImplType::setNumberOfFeatures(nFeatures); }

    //Implementation of decision_forest::classification::Model
    virtual size_t numberOfTrees() const DAAL_C11_OVERRIDE;
    virtual void traverseDF(size_t iTree, classifier::TreeNodeVisitor& visitor) const DAAL_C11_OVERRIDE;
    virtual void traverseBF(size_t iTree, classifier::TreeNodeVisitor& visitor) const DAAL_C11_OVERRIDE;
    virtual void clear() DAAL_C11_OVERRIDE { ImplType::clear(); }

    virtual void traverseDFS(size_t iTree, tree_utils::classification::TreeNodeVisitor& visitor) const DAAL_C11_OVERRIDE;
    virtual void traverseBFS(size_t iTree, tree_utils::classification::TreeNodeVisitor& visitor) const DAAL_C11_OVERRIDE;

    virtual services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE;
    virtual services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE;

    bool add(const TreeType& tree);
};

} // namespace internal
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
