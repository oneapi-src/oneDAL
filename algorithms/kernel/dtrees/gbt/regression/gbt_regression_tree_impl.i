/* file: gbt_regression_tree_impl.i */
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

#ifndef __GBT_REGRESSION_TREE_IMPL_I__
#define __GBT_REGRESSION_TREE_IMPL_I__

#include "gbt_regression_tree_impl.h"

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
services::SharedPtr<AOSNumericTable> TreeTableConnector<algorithmFPType>::createGBTree(size_t maxTreeDepth, services::Status *status)
{
    DAAL_ASSERT(maxTreeDepth >= 0);

    size_t nNodes = (1 << (maxTreeDepth + 1)) - 1;

    services::SharedPtr<AOSNumericTable> table = AOSNumericTable::create(sizeof(TableRecord<algorithmFPType>), 13, nNodes, status);

    if (status && !(*status))
    {
        return services::SharedPtr<AOSNumericTable>();
    }

    table->setFeature<typename TableRecord<algorithmFPType>::FeatureType>  ( 0, DAAL_STRUCT_MEMBER_OFFSET(TableRecord<algorithmFPType>, featureValue    ));
    table->setFeature<int>                                    ( 1, DAAL_STRUCT_MEMBER_OFFSET(TableRecord<algorithmFPType>, featureIdx      ));
    table->setFeature<char>                                   ( 2, DAAL_STRUCT_MEMBER_OFFSET(TableRecord<algorithmFPType>, featureUnordered));
    table->setFeature<typename TableRecord<algorithmFPType>::ResponseType> ( 3, DAAL_STRUCT_MEMBER_OFFSET(TableRecord<algorithmFPType>, response        ));
    table->setFeature<size_t>                                 ( 4, DAAL_STRUCT_MEMBER_OFFSET(TableRecord<algorithmFPType>, level           ));
    table->setFeature<size_t>                                 ( 5, DAAL_STRUCT_MEMBER_OFFSET(TableRecord<algorithmFPType>, nid             ));
    table->setFeature<size_t>                                 ( 6, DAAL_STRUCT_MEMBER_OFFSET(TableRecord<algorithmFPType>, n               ));
    table->setFeature<size_t>                                 ( 7, DAAL_STRUCT_MEMBER_OFFSET(TableRecord<algorithmFPType>, iStart          ));
    table->setFeature<char>                                   ( 8, DAAL_STRUCT_MEMBER_OFFSET(TableRecord<algorithmFPType>, nodeState       ));
    table->setFeature<char>                                   ( 9, DAAL_STRUCT_MEMBER_OFFSET(TableRecord<algorithmFPType>, isFinalized     ));
    table->setFeature<algorithmFPType>                        (10, DAAL_STRUCT_MEMBER_OFFSET(TableRecord<algorithmFPType>, gTotal          ));
    table->setFeature<algorithmFPType>                        (11, DAAL_STRUCT_MEMBER_OFFSET(TableRecord<algorithmFPType>, hTotal          ));
    table->setFeature<size_t>                                 (12, DAAL_STRUCT_MEMBER_OFFSET(TableRecord<algorithmFPType>, nTotal          ));

    table->allocateDataMemory();

    services::internal::service_memset<char, sse2>((char*)table->getArray(), (char)0, sizeof(TableRecord<algorithmFPType>) * nNodes);

    return table;
}

template<typename algorithmFPType>
TreeTableConnector<algorithmFPType>::TreeTableConnector(AOSNumericTable * table): _table(table) {
    _records = (TableRecord<algorithmFPType>*)_table->getArray();
    getSplitLevel(0);
}

template<typename algorithmFPType>
bool TreeTableConnector<algorithmFPType>::getSplitLevel(size_t nid)
{
    DAAL_ASSERT(nid < _table->getNumberOfRows());
    TableRecord<algorithmFPType>& record = _records[nid];

    if (record.nodeState == split)
    {
        if (!record.isFinalized)
        {
            _splitLevel = record.level;
            return true;
        }
        else
        {
            return getSplitLevel(2 * nid + 1) || getSplitLevel(2 * nid + 2);
        }
    }
    return false;
}

template<typename algorithmFPType>
void TreeTableConnector<algorithmFPType>::getSplitNodes(size_t nid, Collection<TableRecord<algorithmFPType> *>& nodesForSplit)
{
    DAAL_ASSERT(nid < _table->getNumberOfRows());
    TableRecord<algorithmFPType>& record = _records[nid];

    if (record.nodeState == split)
    {
        if (!record.isFinalized)
        {
            nodesForSplit << &record;
        }
        else
        {
            TableRecord<algorithmFPType>& leftChild  = _records[2 * nid + 1];
            TableRecord<algorithmFPType>& rightChild = _records[2 * nid + 2];

            if (leftChild.nodeState == split)
            {
                getSplitNodes(2 * nid + 1, nodesForSplit);
            }
            if (rightChild.nodeState == split)
            {
                getSplitNodes(2 * nid + 2, nodesForSplit);
            }
        }
    }
}

template<typename algorithmFPType>
template<CpuType cpu>
void TreeTableConnector<algorithmFPType>::getSplitNodesMerged(size_t nid, Collection<SplitRecord<algorithmFPType>>& nodesForSplit)
{
    DAAL_ASSERT(nid < _table->getNumberOfRows());
    TableRecord<algorithmFPType> & record = _records[nid];

    if (!record.isFinalized)
    {
        SplitRecord<algorithmFPType> splitRecord;
        splitRecord.first = &record;
        nodesForSplit << splitRecord;
    }
    else
    if (record.nodeState != leaf)
    {
        if (record.nodeState == badSplit)
        {
            if (record.level == _splitLevel - 1)
            {
                nodesForSplit << SplitRecord<algorithmFPType>();
            }
        }
        else
        {
            TableRecord<algorithmFPType> & leftChild  = _records[2 * nid + 1];
            TableRecord<algorithmFPType> & rightChild = _records[2 * nid + 2];

            if (record.level == _splitLevel - 1)
            {
                SplitRecord<algorithmFPType> splitRecord;

                if (!leftChild.isFinalized && leftChild.nodeState == split)
                {
                    splitRecord.first = &leftChild;
                }
                if (!rightChild.isFinalized && rightChild.nodeState == split)
                {
                    splitRecord.second = &rightChild;
                }

                nodesForSplit << splitRecord;
            }

            if (leftChild.isFinalized && leftChild.nodeState != leaf)
            {
                getSplitNodesMerged<cpu>(2 * nid + 1, nodesForSplit);
            }
            if (rightChild.isFinalized && rightChild.nodeState != leaf)
            {
                getSplitNodesMerged<cpu>(2 * nid + 2, nodesForSplit);
            }
        }
    }
}

template<typename algorithmFPType>
void TreeTableConnector<algorithmFPType>::getLeafNodes(size_t nid, Collection<TableRecord<algorithmFPType> *>& leaves)
{
    DAAL_ASSERT(nid < _table->getNumberOfRows());
    TableRecord<algorithmFPType>& record = _records[nid];

    if (record.nodeState == split)
    {
        TableRecord<algorithmFPType>& leftChild  = _records[2 * nid + 1];
        TableRecord<algorithmFPType>& rightChild = _records[2 * nid + 2];

        getLeafNodes(2 * nid + 1, leaves);
        getLeafNodes(2 * nid + 2, leaves);
    }
    else
    {
        leaves << &record;
    }
}

template<typename algorithmFPType>
TableRecord<algorithmFPType>* TreeTableConnector<algorithmFPType>::get(size_t nid)
{
    return &(_records[nid]);
}

template<typename algorithmFPType>
void TreeTableConnector<algorithmFPType>::createNode(size_t level, size_t nid, size_t n, size_t iStart, algorithmFPType gTotal, algorithmFPType hTotal, size_t nTotal, const training::Parameter &par)
{
    DAAL_ASSERT(nid < _table->getNumberOfRows());

    TableRecord<algorithmFPType>& record = _records[nid];

    record.level = level;
    record.nid = nid;
    record.n = n;
    record.iStart = iStart;
    record.isFinalized = false;
    record.gTotal = gTotal;
    record.hTotal = hTotal;
    record.nTotal = nTotal;

    if ((nTotal < 2 * par.minObservationsInLeafNode) || ((par.maxTreeDepth > 0) && (level >= par.maxTreeDepth))) // terminate criteria
    {
        record.nodeState = leaf;
    }
    else
    {
        record.nodeState = split;
    }
}

template<typename algorithmFPType>
void TreeTableConnector<algorithmFPType>::getMaxLevel(size_t nid, size_t &maxLevel)
{
    TableRecord<algorithmFPType>& record = _records[nid];

    if (record.level > maxLevel)
    {
        maxLevel = record.level;
    }

    if (record.nodeState == split)
    {
        getMaxLevel(nid * 2 + 1, maxLevel);
        getMaxLevel(nid * 2 + 2, maxLevel);
    }
}

template<typename algorithmFPType>
size_t TreeTableConnector<algorithmFPType>::getNNodes(size_t nid)
{
    TableRecord<algorithmFPType>& record = _records[nid];

    if (record.nodeState == split)
    {
        return 1 + getNNodes(nid * 2 + 1) + getNNodes(nid * 2 + 2);
    }

    return 1;
}

template<typename algorithmFPType>
template<CpuType cpu>
void TreeTableConnector<algorithmFPType>::convertToGbtDecisionTree(algorithmFPType **binValues, const size_t nNodes, const size_t maxLevel,
                                  gbt::internal::GbtDecisionTree *tree, double *impVals, int *nNodeSamplesVals,
                                  const algorithmFPType initialF, const training::Parameter &par)
{
    typedef ImpurityData<algorithmFPType, cpu> ImpurityType;

    services::Collection<TableRecord<algorithmFPType>*> sonsArr(nNodes + 1);
    services::Collection<TableRecord<algorithmFPType>*> parentsArr(nNodes + 1);

    TableRecord<algorithmFPType>** sons = sonsArr.data();
    TableRecord<algorithmFPType>** parents = parentsArr.data();

    gbt::prediction::internal::ModelFPType* const splitPoints = tree->getSplitPoints();
    gbt::prediction::internal::FeatureIndexType* const featureIndexes = tree->getFeatureIndexesForSplit();

    for(size_t i = 0; i < nNodes; ++i)
    {
        sons[i] = nullptr;
        parents[i] = nullptr;
    }

    size_t nParents = 1;
    parents[0] = get(0);
    size_t idxInTable = 0;

    for(size_t level = 0; level < maxLevel + 1; level++)
    {
        size_t nSons = 0;
        for(size_t iParent = 0; iParent < nParents; iParent++)
        {
            TableRecord<algorithmFPType>* p = parents[iParent];

            if(p->nodeState == split)
            {
                sons[nSons++] = get(p->nid * 2 + 1);
                sons[nSons++] = get(p->nid * 2 + 2);
                featureIndexes[idxInTable] = p->featureIdx;
                splitPoints[idxInTable] = binValues[p->featureIdx][p->featureValue];
            }
            else
            {
                if (level < maxLevel)
                {
                    sons[nSons++] = p;
                    sons[nSons++] = p;
                }
                featureIndexes[idxInTable] = 0;
                splitPoints[idxInTable] = initialF + p->response;
            }
            DAAL_ASSERT(featureIndexes[idxInTable] >= 0);
            nNodeSamplesVals[idxInTable] = (int)p->nTotal;
            impVals[idxInTable] = ImpurityType(p->gTotal, p->hTotal).value(par.lambda);

            idxInTable++;
        }

        if (level < maxLevel)
        {
            const size_t size = nSons*sizeof(TableRecord<algorithmFPType>*);
            daal::services::daal_memcpy_s(parents, size, sons, size);
            nParents = nSons;
        }
    }
}

} // namespace internal
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
