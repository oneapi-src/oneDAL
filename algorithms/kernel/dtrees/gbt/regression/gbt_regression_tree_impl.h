/* file: gbt_regression_tree_impl.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

    static services::SharedPtr<AOSNumericTable> createGBTree(size_t maxTreeDepth, services::Status *status = NULL)
    {
        DAAL_ASSERT(maxTreeDepth >= 0);

        size_t nNodes = (1 << (maxTreeDepth + 1)) - 1;

        services::SharedPtr<AOSNumericTable> table = AOSNumericTable::create(sizeof(TableRecordType), 13, nNodes, status);

        if (status && !(*status))
        {
            return services::SharedPtr<AOSNumericTable>();
        }

        table->setFeature<TableRecordType::FeatureType> ( 0, DAAL_STRUCT_MEMBER_OFFSET(TableRecordType, featureValue    ));
        table->setFeature<int>                          ( 1, DAAL_STRUCT_MEMBER_OFFSET(TableRecordType, featureIdx      ));
        table->setFeature<char>                         ( 2, DAAL_STRUCT_MEMBER_OFFSET(TableRecordType, featureUnordered));
        table->setFeature<TableRecordType::ResponseType>( 3, DAAL_STRUCT_MEMBER_OFFSET(TableRecordType, response        ));
        table->setFeature<size_t>                       ( 4, DAAL_STRUCT_MEMBER_OFFSET(TableRecordType, level           ));
        table->setFeature<size_t>                       ( 5, DAAL_STRUCT_MEMBER_OFFSET(TableRecordType, nid             ));
        table->setFeature<size_t>                       ( 6, DAAL_STRUCT_MEMBER_OFFSET(TableRecordType, n               ));
        table->setFeature<size_t>                       ( 7, DAAL_STRUCT_MEMBER_OFFSET(TableRecordType, iStart          ));
        table->setFeature<char>                         ( 8, DAAL_STRUCT_MEMBER_OFFSET(TableRecordType, nodeState       ));
        table->setFeature<char>                         ( 9, DAAL_STRUCT_MEMBER_OFFSET(TableRecordType, isFinalized     ));
        table->setFeature<algorithmFPType>              (10, DAAL_STRUCT_MEMBER_OFFSET(TableRecordType, gTotal          ));
        table->setFeature<algorithmFPType>              (11, DAAL_STRUCT_MEMBER_OFFSET(TableRecordType, hTotal          ));
        table->setFeature<size_t>                       (12, DAAL_STRUCT_MEMBER_OFFSET(TableRecordType, nTotal          ));

        table->allocateDataMemory();

        services::internal::service_memset<char, sse2>((char*)table->getArray(), (char)0, sizeof(TableRecordType) * nNodes);

        return table;
    }

    TreeTableConnector(AOSNumericTable * table): _table(table) {
        _records = (TableRecordType*)_table->getArray();
        getSplitLevel(0);
    }

    bool getSplitLevel(size_t nid)
    {
        DAAL_ASSERT(nid < _table->getNumberOfRows());
        TableRecordType& record = _records[nid];

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

    void getSplitNodes(size_t nid, Collection<TableRecordType *>& nodesForSplit)
    {
        DAAL_ASSERT(nid < _table->getNumberOfRows());
        TableRecordType& record = _records[nid];

        if (record.nodeState == split)
        {
            if (!record.isFinalized)
            {
                nodesForSplit << &record;
            }
            else
            {
                TableRecordType& leftChild  = _records[2 * nid + 1];
                TableRecordType& rightChild = _records[2 * nid + 2];

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

    void getSplitNodesMerged(size_t nid, Collection<SplitRecordType>& nodesForSplit)
    {
        DAAL_ASSERT(nid < _table->getNumberOfRows());
        TableRecordType & record = _records[nid];

        if (!record.isFinalized)
        {
            SplitRecordType splitRecord;
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
                    nodesForSplit << SplitRecordType();
                }
            }
            else
            {
                TableRecordType & leftChild  = _records[2 * nid + 1];
                TableRecordType & rightChild = _records[2 * nid + 2];

                if (record.level == _splitLevel - 1)
                {
                    SplitRecordType splitRecord;

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
                    getSplitNodesMerged(2 * nid + 1, nodesForSplit);
                }
                if (rightChild.isFinalized && rightChild.nodeState != leaf)
                {
                    getSplitNodesMerged(2 * nid + 2, nodesForSplit);
                }
            }
        }
    }

    void getLeafNodes(size_t nid, Collection<TableRecordType *>& leaves)
    {
        DAAL_ASSERT(nid < _table->getNumberOfRows());
        TableRecordType& record = _records[nid];

        if (record.nodeState == split)
        {
            TableRecordType& leftChild  = _records[2 * nid + 1];
            TableRecordType& rightChild = _records[2 * nid + 2];

            getLeafNodes(2 * nid + 1, leaves);
            getLeafNodes(2 * nid + 2, leaves);
        }
        else
        {
            leaves << &record;
        }
    }

    TableRecordType* get(size_t nid)
    {
        return &(_records[nid]);
    }

    void createNode(size_t level, size_t nid, size_t n, size_t iStart, algorithmFPType gTotal, algorithmFPType hTotal, size_t nTotal, const training::Parameter &par)
    {
        DAAL_ASSERT(nid < _table->getNumberOfRows());

        TableRecordType& record = _records[nid];

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

    void getMaxLevel(size_t nid, size_t &maxLevel)
    {
        TableRecordType& record = _records[nid];

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

    size_t getNNodes(size_t nid)
    {
        TableRecordType& record = _records[nid];

        if (record.nodeState == split)
        {
            return 1 + getNNodes(nid * 2 + 1) + getNNodes(nid * 2 + 2);
        }

        return 1;
    }

    template<CpuType cpu>
    void convertToGbtDecisionTree(algorithmFPType **binValues, const size_t nNodes, const size_t maxLevel,
                                  gbt::internal::GbtDecisionTree *tree, double *impVals, int *nNodeSamplesVals,
                                  const algorithmFPType initialF, const training::Parameter &par)
    {
        typedef ImpurityData<algorithmFPType, cpu> ImpurityType;

        services::Collection<TableRecordType*> sonsArr(nNodes + 1);
        services::Collection<TableRecordType*> parentsArr(nNodes + 1);

        TableRecordType** sons = sonsArr.data();
        TableRecordType** parents = parentsArr.data();

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
                TableRecordType* p = parents[iParent];

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
                const size_t size = nSons*sizeof(TableRecordType*);
                daal::services::daal_memcpy_s(parents, size, sons, size);
                nParents = nSons;
            }
        }
    }

    void initialize();

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
