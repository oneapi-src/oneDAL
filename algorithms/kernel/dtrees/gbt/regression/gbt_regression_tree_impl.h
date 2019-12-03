/* file: gbt_regression_tree_impl.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of the class defining the gradient boosted trees tree function
//--
*/

#ifndef __GBT_REGRESSION_TREE_IMPL__
#define __GBT_REGRESSION_TREE_IMPL__

#include "data_management/data/aos_numeric_table.h"
#include "gbt_model_impl.h"
#include "gbt_train_aux.i"

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

using namespace daal::algorithms::dtrees::internal;
using namespace daal::algorithms::gbt::training::internal;

template<typename algorithmFPType>
struct TableRecord
{
    typedef int FeatureType;
    typedef algorithmFPType ResponseType;

    // split fields
    FeatureType featureValue;
    int featureIdx;
    char featureUnordered;

    // leaf fields
    ResponseType response;

    size_t    level;
    size_t    nid;
    size_t    n;
    size_t    iStart;

    char      nodeState;
    char      isFinalized;

    algorithmFPType gTotal;
    algorithmFPType hTotal;
    size_t          nTotal;
};

template<typename algorithmFPType>
struct SplitRecord
{
    // 2 x ptr on record
    SplitRecord(): first(nullptr), second(nullptr) {}

    TableRecord<algorithmFPType> *first;
    TableRecord<algorithmFPType> *second;
};


template<typename algorithmFPType>
class TreeTableConnector
{
    typedef TableRecord<algorithmFPType> TableRecordType;
    typedef SplitRecord<algorithmFPType> SplitRecordType;

public:

    enum NodeState
    {
        leaf,
        split,
        badSplit
    };

    static services::SharedPtr<AOSNumericTable> createGBTree(size_t maxTreeDepth, services::Status *status = NULL);

    TreeTableConnector(AOSNumericTable * table);

    bool getSplitLevel(size_t nid);

    void getSplitNodes(size_t nid, Collection<TableRecordType *>& nodesForSplit);

    template<CpuType cpu>
    void getSplitNodesMerged(size_t nid, Collection<SplitRecordType>& nodesForSplit);

    void getLeafNodes(size_t nid, Collection<TableRecordType *>& leaves);

    TableRecordType* get(size_t nid);

    void createNode(size_t level, size_t nid, size_t n, size_t iStart, algorithmFPType gTotal, algorithmFPType hTotal, size_t nTotal, const training::Parameter &par);

    void getMaxLevel(size_t nid, size_t &maxLevel);

    size_t getNNodes(size_t nid);

    template<CpuType cpu>
    void convertToGbtDecisionTree(algorithmFPType **binValues, const size_t nNodes, const size_t maxLevel,
                                  gbt::internal::GbtDecisionTree *tree, double *impVals, int *nNodeSamplesVals,
                                  const algorithmFPType initialF, const training::Parameter &par);

private:
    AOSNumericTable * _table;
    TableRecordType * _records;
    size_t _splitLevel = 0;
};

} // namespace internal
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
