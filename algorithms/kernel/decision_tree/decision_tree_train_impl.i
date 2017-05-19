/* file: decision_tree_train_impl.i */
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
//  Common functions for Decision tree training
//--
*/

#ifndef __DECISION_TREE_TRAIN_IMPL_I__
#define __DECISION_TREE_TRAIN_IMPL_I__

#include "numeric_table.h"
#include "threading.h"
#include "decision_tree_impl.i"
#include "service_utils.h"
#include "service_sort.h"
#include "service_math.h"
#include "service_data_utils.h"
#include "data_utils.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace internal
{

using namespace daal::data_management;
using namespace daal::services::internal;
using namespace daal::algorithms::internal;

typedef size_t TreeNodeIndex;

template <CpuType cpu, typename IndependentVariable, typename DependentVariable>
class TreeNode
{
public:
    typedef IndependentVariable IndependentVariableType;
    typedef DependentVariable DependentVariableType;

    // Constructs leaf.
    TreeNode(DependentVariableType dependentVariable) : _leftChildIndex(0), _dependentVariable(dependentVariable) {}

    // Constructs decision node.
    TreeNode(FeatureIndex featureIndex, IndependentVariableType cutPoint, TreeNodeIndex leftChildIndex) : _leftChildIndex(leftChildIndex),
                                                                                                          _featureIndex(featureIndex),
                                                                                                          _cutPoint(cutPoint)
    {
        DAAL_ASSERT(leftChildIndex != 0);
    }

    bool isLeaf() const { return (_leftChildIndex == 0); }

    TreeNodeIndex leftChildIndex() const
    {
        DAAL_ASSERT(!isLeaf());
        return _leftChildIndex;
    }

    TreeNodeIndex rightChildIndex() const
    {
        DAAL_ASSERT(!isLeaf());
        return _leftChildIndex + 1;
    }

    FeatureIndex featureIndex() const
    {
        DAAL_ASSERT(!isLeaf());
        return _featureIndex;
    }

    IndependentVariableType cutPoint() const
    {
        DAAL_ASSERT(!isLeaf());
        return _cutPoint;
    }

    DependentVariableType dependentVariable() const
    {
        DAAL_ASSERT(isLeaf());
        return _dependentVariable;
    }

private:
    TreeNodeIndex _leftChildIndex;
    union
    {
        // Decision node.
        struct
        {
            FeatureIndex _featureIndex;
            IndependentVariableType _cutPoint;
        };

        // Leaf node.
        DependentVariableType _dependentVariable;
    };
};

template <CpuType cpu>
struct BaseCutPointFinder
{
    template <typename SplitCriterion, typename RandomIterator, typename GetIndependentValue, typename GetDependentValue, typename Compare>
    static RandomIterator find(SplitCriterion & splitCriterion, RandomIterator first, RandomIterator last,
                               typename SplitCriterion::DataStatistics & dataStatistics,
                               const typename SplitCriterion::DataStatistics & totalDataStatistics,
                               data_management::data_feature_utils::FeatureType featureType, RandomIterator & greater,
                               typename SplitCriterion::ValueType & winnerSplitCriterionValue,
                               typename SplitCriterion::DataStatistics & winnerDataStatistics,
                               GetIndependentValue getIndependentValue, GetDependentValue getDependentValue, Compare compare)
    {
        auto winner = last;
        if (first != last)
        {
            const size_t totalCount = last - first;
            if (featureType != data_management::data_feature_utils::DAAL_CATEGORICAL)
            {
                dataStatistics.reset(totalDataStatistics);
                for (auto i = first;;)
                {
                    const auto next = upperBound<cpu>(i, last, *i, compare);
                    if (next == last) { break; }

                    for (auto j = i; j != next; ++j)
                    {
                        dataStatistics.update(getDependentValue(*j));
                    }

                    const size_t leftCount = next - first;
                    const size_t rightCount = last - next;
                    DAAL_ASSERT(leftCount + rightCount == totalCount);

                    const auto splitCriterionValue = splitCriterion(first, last, i, next, dataStatistics, totalDataStatistics, featureType,
                                                                    leftCount, rightCount, totalCount);

                    if (!(winner != last) || (splitCriterionValue < winnerSplitCriterionValue))
                    {
                        winner = i;
                        greater = next;
                        winnerSplitCriterionValue = splitCriterionValue;
                        winnerDataStatistics = dataStatistics;
                    }

                    i = next;
                }
            }
            else
            {
                auto i = first;
                auto next = upperBound<cpu>(i, last, *i, compare);
                if (next != last)
                {
                    for (;; next = upperBound<cpu>(i, last, *i, compare))
                    {
                        dataStatistics.reset(totalDataStatistics);

                        for (auto j = i; j != next; ++j)
                        {
                            dataStatistics.update(getDependentValue(*j));
                        }

                        const size_t leftCount = next - i;
                        const size_t rightCount = totalCount - leftCount;
                        DAAL_ASSERT(rightCount == (i - first) + (last - next));

                        const auto splitCriterionValue = splitCriterion(first, last, i, next, dataStatistics, totalDataStatistics, featureType,
                                                                        leftCount, rightCount, totalCount);

                        if (!(winner != last) || (splitCriterionValue < winnerSplitCriterionValue))
                        {
                            winner = i;
                            greater = next;
                            winnerSplitCriterionValue = splitCriterionValue;
                            winnerDataStatistics = dataStatistics;
                        }

                        if (next == last) { break; }
                        i = next;
                    }
                }
            }
        }

        return winner;
    }
};

template <CpuType cpu, typename SplitCriterion>
struct CutPointFinder : private BaseCutPointFinder<cpu>
{
    using BaseCutPointFinder<cpu>::find;
};

template <CpuType cpu, typename IndependentVariable, typename DependentVariable>
class Tree
{
public:
    typedef IndependentVariable IndependentVariableType;
    typedef DependentVariable DependentVariableType;
    typedef TreeNode<cpu, IndependentVariableType, DependentVariableType> TreeNodeType;

    Tree() : _nodes(nullptr), _nodeCount(0), _nodeCapacity(0) {}

    Tree(const Tree &) = delete;
    Tree & operator= (const Tree &) = delete;

    ~Tree()
    {
        daal_free(_nodes);
    }

    size_t nodeCount() const { return _nodeCount; }

    const TreeNodeType & operator[] (size_t index) const
    {
        DAAL_ASSERT(index < _nodeCount);
        return _nodes[index];
    }

    TreeNodeType & operator[] (size_t index)
    {
        DAAL_ASSERT(index < _nodeCount);
        return _nodes[index];
    }

    template <typename SplitCriterion>
    void train(SplitCriterion & splitCriterion, const NumericTable & x, const NumericTable & y, size_t numberOfClasses = 0, size_t maxTreeDepth = 0,
               size_t minLeafObservations = 1, size_t minSplitObservations = 2)
    {
        const size_t xRowCount = x.getNumberOfRows();
        FeatureTypesCache featureTypesCache(x);

        typename SplitCriterion::DataStatistics totalDataStatistics(numberOfClasses, x, y), dataStatistics(numberOfClasses);

        size_t * const indexes = prepareIndexes(xRowCount);

        clear();
        internalTrain(splitCriterion, x, y, indexes, xRowCount, pushBack(), featureTypesCache, dataStatistics, totalDataStatistics,
                      maxTreeDepth != 0 ? maxTreeDepth : static_cast<size_t>(-1), minLeafObservations, minSplitObservations);
        daal_free(indexes);
    }

    template <typename Data>
    void reducedErrorPruning(const NumericTable & px, const NumericTable & py, Data & data) const
    {
        typedef typename Data::ErrorType ErrorType;

        if (!empty())
        {
            DAAL_ASSERT(_nodeCount == data.size());

            FeatureTypesCache featureTypesCache(px);

            const size_t pxRowCount = px.getNumberOfRows();
            const size_t pxColumnCount = px.getNumberOfColumns();
            DAAL_ASSERT(pxRowCount == py.getNumberOfRows());
            BlockDescriptor<IndependentVariableType> pxBD;
            BlockDescriptor<DependentVariableType> pyBD;
            const_cast<NumericTable &>(px).getBlockOfRows(0, pxRowCount, readOnly, pxBD);
            const_cast<NumericTable &>(py).getBlockOfColumnValues(0, 0, pxRowCount, readOnly, pyBD);
            const IndependentVariableType * const dpx = pxBD.getBlockPtr();
            const DependentVariableType * const dpy = pyBD.getBlockPtr();

            for (size_t i = 0; i < pxRowCount; ++i)
            {
                const IndependentVariableType * const p = &dpx[i * pxColumnCount];
                size_t nodeIdx = 0;
                DAAL_ASSERT(nodeIdx < _nodeCount);
                while (!_nodes[nodeIdx].isLeaf())
                {
                    data.update(nodeIdx, dpy[i]);
                    const size_t nodeFeatureIndex = _nodes[nodeIdx].featureIndex();
                    switch (featureTypesCache[nodeFeatureIndex])
                    {
                    case data_management::data_feature_utils::DAAL_CATEGORICAL:
                        nodeIdx = (p[nodeFeatureIndex] == _nodes[nodeIdx].cutPoint() ? _nodes[nodeIdx].leftChildIndex()
                                                                                     : _nodes[nodeIdx].rightChildIndex());
                        DAAL_ASSERT(nodeIdx < _nodeCount);
                        break;
                    case data_management::data_feature_utils::DAAL_ORDINAL:
                    case data_management::data_feature_utils::DAAL_CONTINUOUS:
                        nodeIdx = (p[nodeFeatureIndex] < _nodes[nodeIdx].cutPoint() ? _nodes[nodeIdx].leftChildIndex()
                                                                                    : _nodes[nodeIdx].rightChildIndex());
                        DAAL_ASSERT(nodeIdx < _nodeCount);
                        break;
                    default:
                        DAAL_ASSERT(false);
                        break;
                    }
                }

                data.update(nodeIdx, dpy[i]);
            }

            const_cast<NumericTable &>(py).releaseBlockOfColumnValues(pyBD);
            const_cast<NumericTable &>(px).releaseBlockOfRows(pxBD);

            internalREP<ErrorType, Data>(0, data);
        }
    }

    template <typename Functor>
    void enumerateNodes(Functor f) const
    {
        if (!empty())
        {
            internalEnumerateNodes(0, 0, f);
        }
    }

protected:
    void reserve(size_t newCapacity)
    {
        if (newCapacity > _nodeCapacity)
        {
            TreeNodeType * newNodes = daal_alloc<TreeNodeType>(newCapacity);
            daal_memcpy_s(newNodes, newCapacity * sizeof(TreeNodeType), _nodes, _nodeCount * sizeof(TreeNodeType));
            swap<cpu>(_nodes, newNodes);
            swap<cpu>(_nodeCapacity, newCapacity);
            daal_free(newNodes);
        }
    }

    TreeNodeIndex pushBack()
    {
        if (_nodeCount >= _nodeCapacity)
        {
            reserve(max<cpu>(_nodeCount + 1, _nodeCapacity * 2));
        }

        return _nodeCount++;
    }

    void clear()
    {
        _nodeCount = 0;
    }

    bool empty() const { return (_nodeCount == 0); }

    size_t * prepareIndexes(size_t size)
    {
        size_t * const indexes = daal_alloc<size_t>(size);
        for (size_t i = 0; i < size; ++i)
        {
            indexes[i] = i;
        }
        return indexes;
    }

    void makeLeaf(TreeNodeIndex nodeIndex, DependentVariableType dependentVariable)
    {
        const TreeNodeType tmp(dependentVariable);
        _nodes[nodeIndex] = tmp;
        DAAL_ASSERT(_nodes[nodeIndex].isLeaf());
        DAAL_ASSERT(_nodes[nodeIndex].dependentVariable() == dependentVariable);
    }

    void makeSplit(TreeNodeIndex nodeIndex, FeatureIndex featureIndex, IndependentVariable cutPoint)
    {
        const TreeNodeType tmp(featureIndex, cutPoint, pushBack());
        _nodes[nodeIndex] = tmp;
        pushBack();
        DAAL_ASSERT(!_nodes[nodeIndex].isLeaf());
        DAAL_ASSERT(_nodes[nodeIndex].featureIndex() == featureIndex);
        DAAL_ASSERT(_nodes[nodeIndex].cutPoint() == cutPoint);
        DAAL_ASSERT(_nodes[nodeIndex].leftChildIndex() == _nodeCount - 2);
        DAAL_ASSERT(_nodes[nodeIndex].rightChildIndex() == _nodeCount - 1);
    }

    template <typename SplitCriterion>
    void internalTrain(SplitCriterion & splitCriterion, const NumericTable & x, const NumericTable & y, size_t * indexes, size_t indexCount,
                       TreeNodeIndex nodeIndex, const FeatureTypesCache & featureTypesCache,
                       typename SplitCriterion::DataStatistics & dataStatistics,
                       const typename SplitCriterion::DataStatistics & totalDataStatistics, size_t depthLimit, size_t minLeafSize,
                       size_t minSplitSize)
    {
        typedef data_management::BlockDescriptor<IndependentVariableType> IndependentVariableBD;
        typedef data_management::BlockDescriptor<DependentVariableType> DependentVariableBD;
        typedef daal::internal::Math<typename SplitCriterion::ValueType, cpu> SplitCriterionMath;

        DAAL_ASSERT(depthLimit != 0);
        DAAL_ASSERT(minLeafSize >= 1);
        DAAL_ASSERT(indexes);

        if (depthLimit == 1 || indexCount < minSplitSize || indexCount < minLeafSize * 2)
        {
            makeLeaf(nodeIndex, totalDataStatistics.getBestDependentVariableValue());
            return;
        }

        {
            typename SplitCriterion::DependentVariableType leafDependentVariableValue;
            if (totalDataStatistics.isPure(leafDependentVariableValue))
            {
                makeLeaf(nodeIndex, leafDependentVariableValue);
                return;
            }
        }

        FeatureIndex winnerFeatureIndex = 0;
        IndependentVariableType winnerCutPoint;
        typename SplitCriterion::ValueType winnerSplitCriterionValue, splitCriterionValue;
        size_t winnerPointsAtLeft;
        typename SplitCriterion::DataStatistics winnerDataStatistics, bestCutPointDataStatistics;
        bool winnerIsLeaf = true;

        const size_t featureCount = x.getNumberOfColumns();

        struct Item
        {
            IndependentVariable x;
            DependentVariable y;
        };

        struct Local
        {
            FeatureIndex winnerFeatureIndex;
            IndependentVariableType winnerCutPoint;
            typename SplitCriterion::ValueType winnerSplitCriterionValue, splitCriterionValue;
            size_t winnerPointsAtLeft;
            typename SplitCriterion::DataStatistics winnerDataStatistics, bestCutPointDataStatistics, dataStatistics;
            bool winnerIsLeaf;

            Local() : winnerIsLeaf(true) {}
        };

        daal::tls<Local *> localTLS([=]()-> Local *
        {
            Local * const ptr = new Local;
            return ptr;
        } );

        const typename SplitCriterion::ValueType epsilon = daal::data_feature_utils::internal::EpsilonVal<typename SplitCriterion::ValueType, cpu>::get();

        daal::threader_for(featureCount, featureCount, [=, &localTLS, &indexes, &splitCriterion, &totalDataStatistics,
                                                        &featureTypesCache, &x, &y](int featureIndex)
        {
            Local * const local = localTLS.local();

            Item * const items = daal_alloc<Item>(indexCount);
            Item * const items2 = daal_alloc<Item>(indexCount);

            IndependentVariableBD xBD;
            DependentVariableBD yBD;
            for (size_t i = 0; i < indexCount; ++i)
            {
                const_cast<NumericTable &>(x).getBlockOfColumnValues(featureIndex, indexes[i], 1, data_management::readOnly, xBD);
                const IndependentVariableType * const dx = xBD.getBlockPtr();
                const_cast<NumericTable &>(y).getBlockOfRows(indexes[i], 1, data_management::readOnly, yBD);
                const DependentVariableType * const dy = yBD.getBlockPtr();

                items[i].x = *dx;
                items[i].y = *dy;

                const_cast<NumericTable &>(x).releaseBlockOfColumnValues(xBD);
                const_cast<NumericTable &>(x).releaseBlockOfRows(yBD);
            }

            radixSort<cpu, IndependentVariableType>(items, indexCount, items2, [](const Item & v) -> const IndependentVariableType &
            {
                return v.x;
            });

            Item * next = nullptr;
            const auto i = CutPointFinder<cpu, SplitCriterion>::find(splitCriterion, items, &items[indexCount], local->dataStatistics,
                                                                     totalDataStatistics,
                                                                     featureTypesCache[featureIndex], next, local->splitCriterionValue,
                                                                     local->bestCutPointDataStatistics,
                                                                     [](const Item & v) -> IndependentVariableType { return v.x; },
                                                                     [](const Item & v) -> DependentVariable { return v.y; },
                                                                     [](const Item & v1, const Item & v2) -> bool { return v1.x < v2.x; });

            if (i != &items[indexCount] && (local->winnerIsLeaf || local->splitCriterionValue < local->winnerSplitCriterionValue ||
                                            (SplitCriterionMath::sFabs(local->splitCriterionValue - local->winnerSplitCriterionValue) <= epsilon &&
                                             local->winnerFeatureIndex > featureIndex)))
            {
                local->winnerIsLeaf = false;
                local->winnerFeatureIndex = featureIndex;
                local->winnerSplitCriterionValue = local->splitCriterionValue;
                switch (featureTypesCache[featureIndex])
                {
                case data_management::data_feature_utils::DAAL_CATEGORICAL:
                    local->winnerCutPoint = i->x;
                    break;
                case data_management::data_feature_utils::DAAL_ORDINAL:
                    local->winnerCutPoint = next->x;
                    break;
                case data_management::data_feature_utils::DAAL_CONTINUOUS:
                    local->winnerCutPoint = (i->x + next->x) / 2;
                    break;
                default:
                    DAAL_ASSERT(false);
                    break;
                }
                local->winnerPointsAtLeft = next - items; // distance.
                local->winnerDataStatistics = local->bestCutPointDataStatistics;
            }

            daal_free(items2);
            daal_free(items);
        } );

        localTLS.reduce([=, &winnerIsLeaf, &winnerSplitCriterionValue, &winnerFeatureIndex, &winnerCutPoint, &winnerPointsAtLeft,
                         &winnerDataStatistics](Local * v) -> void
        {
            if ((!v->winnerIsLeaf) && (winnerIsLeaf || v->winnerSplitCriterionValue < winnerSplitCriterionValue ||
                                       (SplitCriterionMath::sFabs(winnerSplitCriterionValue - v->winnerSplitCriterionValue) <= epsilon &&
                                        winnerFeatureIndex > v->winnerFeatureIndex)))
            {
                winnerIsLeaf = false;
                winnerFeatureIndex = v->winnerFeatureIndex;
                winnerSplitCriterionValue = v->winnerSplitCriterionValue;
                winnerCutPoint = v->winnerCutPoint;
                winnerPointsAtLeft = v->winnerPointsAtLeft;
                winnerDataStatistics = v->winnerDataStatistics;
            }

            delete v;
        } );

        if (winnerIsLeaf || winnerPointsAtLeft < minLeafSize || indexCount - winnerPointsAtLeft < minLeafSize)
        {
            makeLeaf(nodeIndex, totalDataStatistics.getBestDependentVariableValue());
            return;
        }

        makeSplit(nodeIndex, winnerFeatureIndex, winnerCutPoint);
        DAAL_ASSERT(!_nodes[nodeIndex].isLeaf());

        // Partition.
        size_t * splitIndexes = nullptr;
        switch (featureTypesCache[winnerFeatureIndex])
        {
        case data_management::data_feature_utils::DAAL_CATEGORICAL:
            splitIndexes = partition<cpu>(indexes, &indexes[indexCount], [winnerFeatureIndex, winnerCutPoint, &x](size_t i) -> bool
            {
                IndependentVariableBD iBD;
                const_cast<NumericTable &>(x).getBlockOfColumnValues(winnerFeatureIndex, i, 1, data_management::readOnly, iBD);
                const auto v = *(iBD.getBlockPtr());
                const_cast<NumericTable &>(x).releaseBlockOfColumnValues(iBD);
                return (v == winnerCutPoint);
            });

            break;
        case data_management::data_feature_utils::DAAL_ORDINAL:
        case data_management::data_feature_utils::DAAL_CONTINUOUS:
            splitIndexes = partition<cpu>(indexes, &indexes[indexCount], [winnerFeatureIndex, winnerCutPoint, &x](size_t i) -> bool
            {
                IndependentVariableBD iBD;
                const_cast<NumericTable &>(x).getBlockOfColumnValues(winnerFeatureIndex, i, 1, data_management::readOnly, iBD);
                const auto v = *(iBD.getBlockPtr());
                const_cast<NumericTable &>(x).releaseBlockOfColumnValues(iBD);
                return (v < winnerCutPoint);
            });
            break;
        default:
            DAAL_ASSERT(false);
            break;
        }

        // Estimate data statistics after partitioning.
        typename SplitCriterion::DataStatistics & leftDataStatistics = winnerDataStatistics;
        typename SplitCriterion::DataStatistics rightDataStatistics(totalDataStatistics);
        rightDataStatistics -= leftDataStatistics;

        // Process left child.
        internalTrain(splitCriterion, x, y, indexes, splitIndexes - indexes, _nodes[nodeIndex].leftChildIndex(),
                      featureTypesCache, dataStatistics, leftDataStatistics, depthLimit - 1, minLeafSize, minSplitSize);

        // Process right child.
        internalTrain(splitCriterion, x, y, splitIndexes, &indexes[indexCount] - splitIndexes, _nodes[nodeIndex].rightChildIndex(),
                      featureTypesCache, dataStatistics, rightDataStatistics, depthLimit - 1, minLeafSize, minSplitSize);
    }

    template <typename Error, typename Data>
    Error internalREP(size_t nodeIndex, Data & data) const
    {
        if (_nodes[nodeIndex].isLeaf())
        {
            return data.error(nodeIndex, _nodes[nodeIndex].dependentVariable());
        }

        const Error subTreeError = internalREP<Error, Data>(_nodes[nodeIndex].leftChildIndex(), data)
                                   + internalREP<Error, Data>(_nodes[nodeIndex].rightChildIndex(), data);
        const Error leafError = data.error(nodeIndex);
        if (leafError <= subTreeError)
        { // Node must be pruned.
            data.prune(nodeIndex);
            return leafError;
        }
        else
        {
            return subTreeError;
        }
    }

    template <typename Functor>
    void internalEnumerateNodes(TreeNodeIndex nodeIndex, size_t depth, Functor f) const
    {
        f(_nodes[nodeIndex], nodeIndex, depth);
        if (!_nodes[nodeIndex].isLeaf())
        {
            internalEnumerateNodes(_nodes[nodeIndex].leftChildIndex(), depth + 1, f);
            internalEnumerateNodes(_nodes[nodeIndex].rightChildIndex(), depth + 1, f);
        }
    }

private:
    TreeNodeType * _nodes;
    size_t _nodeCount;
    size_t _nodeCapacity;
};

template <CpuType cpu, typename DependentVariable>
class PruningData
{
public:
    typedef DependentVariable DependentVariableType;

    PruningData(size_t size) : _size(size),
                               _dependentVariables(daal_alloc<DependentVariableType>(size ? size : 1)),
                               _isPrunedValues(daal_alloc<bool>(size ? size : 1))
    {
        reset();
    }

    PruningData(const PruningData &) = delete;
    PruningData & operator= (const PruningData &) = delete;

    ~PruningData()
    {
        daal_free(_isPrunedValues);
        daal_free(_dependentVariables);
    }

    size_t size() const { return _size; }

    bool isPruned(size_t index) const
    {
        DAAL_ASSERT(index < _size);
        return _isPrunedValues[index];
    }

    DependentVariableType dependentVariable(size_t index) const
    {
        DAAL_ASSERT(index < _size);
        return _dependentVariables[index];
    }

protected:
    void reset()
    {
        for (size_t i = 0; i < _size; ++i)
        {
            _dependentVariables[i] = 0;
            _isPrunedValues[i] = false;
        }
    }

    void prune(size_t index, DependentVariableType dependentVariable)
    {
        DAAL_ASSERT(index < _size);
        DAAL_ASSERT(!_isPrunedValues[index]);
        _isPrunedValues[index] = true;
        _dependentVariables[index] = dependentVariable;
    }

private:
    size_t _size;
    bool * _isPrunedValues;
    DependentVariableType * _dependentVariables;
};

template <CpuType cpu, typename IndependentVariable, typename DependentVariable>
size_t countNodes(size_t nodeIndex, const Tree<cpu, IndependentVariable, DependentVariable> & tree,
                  const PruningData<cpu, DependentVariable> & pruningData)
{
    if (tree[nodeIndex].isLeaf() || pruningData.isPruned(nodeIndex)) { return 1; }
    return 1 + countNodes<cpu>(tree[nodeIndex].leftChildIndex(), tree, pruningData)
             + countNodes<cpu>(tree[nodeIndex].rightChildIndex(), tree, pruningData);
}

} // namespace internal
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
