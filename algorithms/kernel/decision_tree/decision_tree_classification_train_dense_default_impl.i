/* file: decision_tree_classification_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for Decision tree dense default method.
//--
*/

#ifndef __DECISION_TREE_CLASSIFICATION_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __DECISION_TREE_CLASSIFICATION_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "daal_defines.h"
#include "service_math.h"
#include "numeric_table.h"
#include "decision_tree_classification_model_impl.h"
#include "decision_tree_classification_train_kernel.h"
#include "decision_tree_train_impl.i"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace training
{
namespace internal
{

using namespace daal::data_management;
using namespace daal::internal;
using namespace decision_tree::internal;

template <CpuType cpu>
class ClassCounters
{
public:
    ClassCounters() : _size(0), _counters(nullptr) {}

    ClassCounters(const ClassCounters & value) : _size(value._size), _counters(value._size ? daal_alloc<size_t>(value._size) : nullptr)
    {
        daal_memcpy_s(_counters, _size * sizeof(size_t), value._counters, value._size * sizeof(size_t));
    }

    ClassCounters(size_t size) : _size(size), _counters(size ? daal_alloc<size_t>(size) : nullptr) { reset(); }

    ClassCounters(size_t size, const NumericTable & x, const NumericTable & y) : _size(size), _counters(size ? daal_alloc<size_t>(size) : nullptr)
    {
        if (size)
        {
            reset();
            const size_t yRowCount = y.getNumberOfRows();
            BlockDescriptor<int> yBD;
            const_cast<NumericTable &>(y).getBlockOfColumnValues(0, 0, yRowCount, readOnly, yBD);
            const int * const dy = yBD.getBlockPtr();
            for (size_t i = 0; i < yRowCount; ++i)
            {
                update(static_cast<size_t>(dy[i]));
            }
            const_cast<NumericTable &>(y).releaseBlockOfColumnValues(yBD);
        }
    }

    ~ClassCounters()
    {
        daal_free(_counters);
    }

    ClassCounters & operator= (ClassCounters rhs)
    {
        swap(rhs);
        return *this;
    }

    ClassCounters & operator-= (const ClassCounters & rhs)
    {
        DAAL_ASSERT(_size == rhs._size);
        for (size_t i = 0; i < _size; ++i)
        {
            DAAL_ASSERT(_counters[i] >= rhs._counters[i]);
            _counters[i] -= rhs._counters[i];
        }
        return *this;
    }

    size_t operator[] (size_t index) const
    {
        DAAL_ASSERT(index < _size);
        return _counters[index];
    }

    void swap(ClassCounters & value)
    {
        decision_tree::internal::swap<cpu>(_counters, value._counters);
        decision_tree::internal::swap<cpu>(_size, value._size);
    }

    size_t getBestDependentVariableValue() const
    {
        return maxElement<cpu>(_counters, &_counters[_size]) - _counters;
    }

    void reset(const ClassCounters & value)
    {
        if (_size != value._size)
        {
            _size = value._size;
            size_t * const saveCounters = _counters;
            _counters = _size ? daal_alloc<size_t>(_size) : nullptr;
            daal_free(saveCounters);
        }
        reset();
    }

    void reset()
    {
        for (size_t i = 0; i < _size; ++i) { _counters[i] = 0; }
    }

    void update(size_t index)
    {
        DAAL_ASSERT(index < _size);
        ++_counters[index];
    }

    bool isPure(size_t & onlyClass) const
    {
        size_t numberOfClasses = 0;
        for (size_t i = 0; i < _size; ++i)
        {
            if (_counters[i])
            {
                ++numberOfClasses;
                if (numberOfClasses >= 2) { break; }
                onlyClass = i;
            }
        }
        return (numberOfClasses == 1);
    }

    size_t size() const { return _size; }

private:
    size_t _size;
    size_t * _counters;
};

template <typename algorithmFPType, CpuType cpu>
struct Gini
{
    typedef ClassCounters<cpu> DataStatistics;
    typedef algorithmFPType ValueType;
    typedef size_t DependentVariableType;

    template <typename RandomIterator>
    ValueType operator() (RandomIterator first, RandomIterator last, RandomIterator current, RandomIterator next, DataStatistics & dataStatistics,
                          const DataStatistics & totalDataStatistics, data_management::data_feature_utils::FeatureType featureType,
                          size_t leftCount, size_t rightCount, size_t totalCount)
    {
        const ValueType leftProbability = leftCount * static_cast<ValueType>(1) / totalCount;
        const ValueType rightProbability = rightCount * static_cast<ValueType>(1) / totalCount;
        ValueType leftGini = 1;
        ValueType rightGini = 1;
        const size_t size = dataStatistics.size();
        DAAL_ASSERT(size == totalDataStatistics.size());
        for (size_t i = 0; i < size; ++i)
        {
            const ValueType leftP = dataStatistics[i] * static_cast<ValueType>(1) / leftCount;
            leftGini -= leftP * leftP;
            const ValueType rightP = (totalDataStatistics[i] - dataStatistics[i]) * static_cast<ValueType>(1) / rightCount;
            rightGini -= rightP * rightP;
        }
        return leftProbability * leftGini + rightProbability * rightGini;
    }
};

template <typename algorithmFPType, CpuType cpu>
struct InfoGain
{
    typedef ClassCounters<cpu> DataStatistics;
    typedef algorithmFPType ValueType;
    typedef size_t DependentVariableType;

    template <typename RandomIterator>
    ValueType operator() (RandomIterator first, RandomIterator last, RandomIterator current, RandomIterator next, DataStatistics & dataStatistics,
                          const DataStatistics & totalDataStatistics, data_management::data_feature_utils::FeatureType featureType,
                          size_t leftCount, size_t rightCount, size_t totalCount)
    {
        typedef Math<algorithmFPType, cpu> MathType;

        const ValueType leftProbability = leftCount * static_cast<ValueType>(1) / totalCount;
        const ValueType rightProbability = rightCount * static_cast<ValueType>(1) / totalCount;
        ValueType leftInfo = 0;
        ValueType rightInfo = 0;
        const size_t size = dataStatistics.size();
        DAAL_ASSERT(size == totalDataStatistics.size());
        for (size_t i = 0; i < size; ++i)
        {
            const ValueType leftP = dataStatistics[i] * static_cast<ValueType>(1) / leftCount;
            leftInfo -= (leftP != 0) ? leftP * MathType::sLog(leftP) : 0;
            const ValueType rightP = (totalDataStatistics[i] - dataStatistics[i]) * static_cast<ValueType>(1) / rightCount;
            rightInfo -= (rightP != 0) ? rightP * MathType::sLog(rightP) : 0;
        }
        return leftProbability * leftInfo + rightProbability * rightInfo;
    }
};

template <CpuType cpu>
class REPPruningData : public PruningData<cpu, int>
{
    typedef PruningData<cpu, int> BaseType;

public:
    REPPruningData(size_t size, size_t numberOfClasses) : BaseType(size),
                                                          _numberOfClasses(numberOfClasses),
                                                          _counters(daal_alloc<size_t>(max<cpu, size_t>(1, size * numberOfClasses)))
    {
        resetCounters();
    }

    ~REPPruningData()
    {
        daal_free(_counters);
    }

    void update(size_t index, size_t classIndex)
    {
        DAAL_ASSERT(index < size());
        DAAL_ASSERT(classIndex < _numberOfClasses);
        ++_counters[index * _numberOfClasses + classIndex];
    }

    typedef size_t ErrorType;

    ErrorType error(size_t index, size_t classIndex) const
    {
        DAAL_ASSERT(index < size());
        DAAL_ASSERT(classIndex < _numberOfClasses);

        const size_t * const c = &_counters[index * _numberOfClasses];

        ErrorType err = 0;
        for (size_t i = 0; i < _numberOfClasses; ++i)
        {
            err += c[i];
        }

        return err - c[classIndex];
    }

    ErrorType error(size_t index) const
    {
        DAAL_ASSERT(index < size());

        const size_t * const c = &_counters[index * _numberOfClasses];

        ErrorType err = 0, max = 0;
        for (size_t i = 0; i < _numberOfClasses; ++i)
        {
            err += c[i];

            if (c[i] > max)
            {
                max = c[i];
            }
        }

        return err - max;
    }

    void prune(size_t index)
    {
        BaseType::prune(index, majorityClass(index));
    }

    using BaseType::size;

protected:
    void resetCounters()
    {
        const size_t cnt = size() * _numberOfClasses;
        for (size_t i = 0; i < cnt; ++i)
        {
            _counters[i] = 0;
        }
    }

    size_t majorityClass(size_t index) const
    {
        DAAL_ASSERT(index < size());
        const size_t * const c = &_counters[index * _numberOfClasses];
        return maxElement<cpu>(c, &c[_numberOfClasses]) - c;
    }

private:
    size_t _numberOfClasses;
    size_t * _counters;
};

template <typename algorithmFPType, CpuType cpu>
static void copyNode(size_t srcNodeIdx, size_t destNodeIdx, const decision_tree::internal::Tree<cpu, algorithmFPType, int> & src,
             decision_tree::classification::DecisionTreeNode * dest, size_t & destNodeCount, size_t destNodeCapacity,
             const decision_tree::internal::PruningData<cpu, int> & pruningData)
{
    if (src[srcNodeIdx].isLeaf())
    {
        dest[destNodeIdx].dimension = static_cast<size_t>(-1);
        dest[destNodeIdx].leftIndexOrClass = src[srcNodeIdx].dependentVariable();
        dest[destNodeIdx].cutPoint = 0;
    }
    else if (pruningData.isPruned(srcNodeIdx))
    {
        dest[destNodeIdx].dimension = static_cast<size_t>(-1);
        dest[destNodeIdx].leftIndexOrClass = pruningData.dependentVariable(srcNodeIdx);
        dest[destNodeIdx].cutPoint = 0;
    }
    else
    {
        DAAL_ASSERT(destNodeCount + 2 <= destNodeCapacity);
        dest[destNodeIdx].dimension = src[srcNodeIdx].featureIndex();
        const size_t childIndex = destNodeCount;
        dest[destNodeIdx].leftIndexOrClass = childIndex;
        dest[destNodeIdx].cutPoint = src[srcNodeIdx].cutPoint();
        destNodeCount += 2;
        copyNode(src[srcNodeIdx].leftChildIndex(), childIndex, src, dest, destNodeCount, destNodeCapacity, pruningData);
        copyNode(src[srcNodeIdx].rightChildIndex(), childIndex + 1, src, dest, destNodeCount, destNodeCapacity, pruningData);
    }
}

template <typename algorithmFPType, CpuType cpu>
services::Status DecisionTreeTrainBatchKernel<algorithmFPType, training::defaultDense, cpu>::
    compute(const NumericTable * x, const NumericTable * y, const NumericTable * px, const NumericTable * py,
            decision_tree::classification::Model * r, const daal::algorithms::Parameter * par)
{
    DAAL_ASSERT(x);
    DAAL_ASSERT(y);
    DAAL_ASSERT(r);
    const decision_tree::classification::Parameter * const parameter = static_cast<const decision_tree::classification::Parameter *>(par);
    DAAL_ASSERT(parameter);

    r->setNFeatures(x->getNumberOfColumns());

    Tree<cpu, algorithmFPType, int> tree;
    if (parameter->splitCriterion == gini)
    {
        Gini<algorithmFPType, cpu> splitCriterion;
        tree.train(splitCriterion, *x, *y, parameter->nClasses, parameter->maxTreeDepth, parameter->minObservationsInLeafNodes);
    }
    else
    {
        DAAL_ASSERT(parameter->splitCriterion == infoGain);
        InfoGain<algorithmFPType, cpu> splitCriterion;
        tree.train(splitCriterion, *x, *y, parameter->nClasses, parameter->maxTreeDepth, parameter->minObservationsInLeafNodes);
    }
    if (parameter->pruning == reducedErrorPruning)
    {
        DAAL_ASSERT(px);
        DAAL_ASSERT(py);
        REPPruningData<cpu> repData(tree.nodeCount(), parameter->nClasses);
        tree.reducedErrorPruning(*px, *py, repData);

        const size_t nodeCapacity = countNodes(0, tree, repData);
        DecisionTreeTablePtr treeTable(new DecisionTreeTable(nodeCapacity));
        DecisionTreeNode * const nodes = static_cast<DecisionTreeNode *>(treeTable->getArray());
        size_t nodeCount = 1;
        copyNode<>(0, 0, tree, nodes, nodeCount, nodeCapacity, repData);
        r->impl()->setTreeTable(treeTable);
    }
    else
    {
        const size_t nodeCount = tree.nodeCount();
        DecisionTreeTablePtr treeTable(new DecisionTreeTable(nodeCount));
        DecisionTreeNode * const nodes = static_cast<DecisionTreeNode *>(treeTable->getArray());
        for (size_t i = 0; i < nodeCount; ++i)
        {
            if (tree[i].isLeaf())
            {
                nodes[i].dimension = static_cast<size_t>(-1);
                nodes[i].leftIndexOrClass = tree[i].dependentVariable();
                nodes[i].cutPoint = 0;
            }
            else
            {
                nodes[i].dimension = tree[i].featureIndex();
                nodes[i].leftIndexOrClass = tree[i].leftChildIndex();
                nodes[i].cutPoint = tree[i].cutPoint();
            }
        }
        r->impl()->setTreeTable(treeTable);
    }
    DAAL_RETURN_STATUS();
}

} // namespace internal
} // namespace training
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
