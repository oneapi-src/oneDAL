/* file: decision_tree_classification_model_impl.h */
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
//  Implementation of the class defining the Decision tree model
//--
*/

#ifndef __DECISION_TREE_CLASSIFICATION_MODEL_IMPL_
#define __DECISION_TREE_CLASSIFICATION_MODEL_IMPL_

#include "algorithms/decision_tree/decision_tree_classification_model.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace interface1
{

struct DecisionTreeNode
{
    size_t dimension;
    size_t leftIndexOrClass;
    double cutPoint;
};

class DecisionTreeTable : public data_management::AOSNumericTable
{
public:
    DecisionTreeTable(size_t rowCount = 0) : data_management::AOSNumericTable(sizeof(DecisionTreeNode), 3, rowCount)
    {
        setFeature<size_t> (0, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, dimension));
        setFeature<size_t> (1, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, leftIndexOrClass));
        setFeature<double> (2, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, cutPoint));
        allocateDataMemory();
    }
};

typedef services::SharedPtr<DecisionTreeTable> DecisionTreeTablePtr;
typedef services::SharedPtr<const DecisionTreeTable> DecisionTreeTableConstPtr;

class Model::ModelImpl
{
public:
    /**
     * Empty constructor for deserialization
     */
    ModelImpl() : _TreeTable() {}

    /**
     * Returns the decision tree table
     * \return decision tree table
     */
    DecisionTreeTablePtr getTreeTable() { return _TreeTable; }

    /**
     * Returns the decision tree table
     * \return decision tree table
     */
    DecisionTreeTableConstPtr getTreeTable() const { return _TreeTable; }

    /**
    *  Sets a decision tree table
    *  \param[in]  value  decision tree table
    */
    void setTreeTable(const DecisionTreeTablePtr & value) { _TreeTable = value; }

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive * arch)
    {
        arch->setSharedPtrObj(_TreeTable);
    }

private:
    DecisionTreeTablePtr _TreeTable;
};

} // namespace interface1

using interface1::DecisionTreeTable;
using interface1::DecisionTreeNode;
using interface1::DecisionTreeTablePtr;
using interface1::DecisionTreeTableConstPtr;

} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
