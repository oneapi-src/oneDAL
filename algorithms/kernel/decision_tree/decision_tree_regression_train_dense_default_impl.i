/* file: decision_tree_regression_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for Decision tree dense default method.
//--
*/

#ifndef __DECISION_TREE_REGRESSION_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __DECISION_TREE_REGRESSION_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "daal_defines.h"
#include "service_data_utils.h"
#include "service_math.h"
#include "service_utils.h"
#include "numeric_table.h"
#include "decision_tree_regression_model_impl.h"
#include "decision_tree_regression_train_kernel.h"
#include "decision_tree_train_impl.i"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace training
{
namespace internal
{

using namespace daal::data_management;
using namespace daal::internal;
using namespace decision_tree::internal;

template <typename algorithmFPType, CpuType cpu>
class MSEDataStatistics
{
public:
    MSEDataStatistics() : _mean(0), _count(0), _mse(0) {}

    MSEDataStatistics(size_t) : _mean(0), _count(0), _mse(0) {}

    MSEDataStatistics(size_t, const NumericTable & x, const NumericTable & y) : _mean(0), _count(0), _mse(0)
    {
        const size_t yRowCount = y.getNumberOfRows();
        BlockDescriptor<algorithmFPType> yBD;
        const_cast<NumericTable &>(y).getBlockOfColumnValues(0, 0, yRowCount, readOnly, yBD);
        const algorithmFPType * const dy = yBD.getBlockPtr();
        for (size_t i = 0; i < yRowCount; ++i)
        {
            update(dy[i]);
        }
        const_cast<NumericTable &>(y).releaseBlockOfColumnValues(yBD);
    }

    static algorithmFPType subtractMean(algorithmFPType nab, algorithmFPType mab, algorithmFPType na, algorithmFPType nb, algorithmFPType ma)
    {
        const algorithmFPType mb = (nab * mab - na * ma) / nb;
        return mb;
    }

    static algorithmFPType subtractMSE(algorithmFPType vab, algorithmFPType va, algorithmFPType nab, algorithmFPType mab, algorithmFPType na,
                                       algorithmFPType nb, algorithmFPType ma)
    {
        const algorithmFPType delta = subtractMean(nab, mab, na, nb, ma) - ma;
        const algorithmFPType vb = vab - va - delta * delta * na * nb / nab;
        return vb;
    }

    MSEDataStatistics & operator-= (const MSEDataStatistics & rhs)
    {
        const algorithmFPType newCount = _count - rhs._count;
        const algorithmFPType newMean = subtractMean(_count, _mean, rhs._count, newCount, rhs._mean);
        _mse = subtractMSE(_mse, rhs._mse, _count, _mean, rhs._count, newCount, rhs._mean);
        _count = newCount;
        _mean = newMean;
        return *this;
    }

    algorithmFPType getBestDependentVariableValue() const
    {
        return _mean;
    }

    void reset(const MSEDataStatistics &)
    {
        _mean = algorithmFPType(0);
        _count = algorithmFPType(0);
        _mse = algorithmFPType(0);
    }

    void update(algorithmFPType v)
    {
        // Welford running method.

        if (++_count == 1)
        {
            _mean = v;
            _mse = algorithmFPType(0);
        }
        else
        {
            const algorithmFPType delta = v - _mean;
            _mean += delta / _count;
            _mse += delta * (v - _mean);
        }
    }

    bool isPure(algorithmFPType & result) const
    {
        const algorithmFPType epsilon = daal::services::internal::EpsilonVal<algorithmFPType>::get();
        if (_mse <= epsilon)
        {
            result = _mean;
            return true;
        }

        return false;
    }

    algorithmFPType mean() const { return _mean; }

    algorithmFPType count() const { return _count; }

    algorithmFPType mse() const { return _mse; }

    void swap(MSEDataStatistics & value)
    {
        services::internal::swap<cpu>(_mean, value._mean);
        services::internal::swap<cpu>(_count, value._count);
        services::internal::swap<cpu>(_mse, value._mse);
    }

private:
    algorithmFPType _mean;
    algorithmFPType _count;
    algorithmFPType _mse;
};

template <typename algorithmFPType, CpuType cpu>
struct MSE
{
    typedef MSEDataStatistics<algorithmFPType, cpu> DataStatistics;
    typedef algorithmFPType ValueType;
    typedef algorithmFPType DependentVariableType;

    template <typename RandomIterator>
    ValueType operator() (RandomIterator first, RandomIterator last, RandomIterator current, RandomIterator next, DataStatistics & dataStatistics,
                          const DataStatistics & totalDataStatistics, data_management::features::FeatureType featureType,
                          size_t leftCount, size_t rightCount, size_t totalCount)
    {
        const ValueType leftMSE = dataStatistics.mse();
        const ValueType rightMSE = DataStatistics::subtractMSE(totalDataStatistics.mse(), leftMSE, totalCount, totalDataStatistics.mean(),
                                                               leftCount, rightCount, dataStatistics.mean());
        return leftMSE + rightMSE;
    }

    ValueType operator() (const DataStatistics & totalDataStatistics, size_t totalCount)
    {
        return totalDataStatistics.mse()/totalCount;
    }
};

template <typename algorithmFPType, CpuType cpu>
class REPPruningData : public PruningData<cpu, algorithmFPType>
{
    typedef PruningData<cpu, algorithmFPType> BaseType;

public:
    REPPruningData(size_t size) : BaseType(size), _data(daal_alloc<algorithmFPType>(size ? size * 3 : 1))
    {
        resetData();
    }

    ~REPPruningData()
    {
        daal_free(_data);
    }

    void update(size_t index, algorithmFPType v)
    {
        DAAL_ASSERT(index < size());
        algorithmFPType & count = _data[index * 3];
        algorithmFPType & mean = _data[index * 3 + 1];
        algorithmFPType & mse = _data[index * 3 + 2];

        // Welford running method.

        if (++count == 1)
        {
            mean = v;
            mse = algorithmFPType(0);
        }
        else
        {
            const algorithmFPType delta = v - mean;
            mean += delta / count;
            mse += delta * (v - mean);
        }
    }

    algorithmFPType count(size_t index) const
    {
        DAAL_ASSERT(index < size());
        return _data[index * 3];
    }

    algorithmFPType mean(size_t index) const
    {
        DAAL_ASSERT(index < size());
        return _data[index * 3 + 1];
    }

    algorithmFPType mse(size_t index) const
    {
        DAAL_ASSERT(index < size());
        return _data[index * 3 + 2];
    }

    typedef algorithmFPType ErrorType;

    ErrorType error(size_t index, algorithmFPType v) const
    {
        const algorithmFPType delta = mean(index) - v;
        return mse(index) + count(index) * delta * delta;
    }

    ErrorType error(size_t index) const
    {
        return mse(index);
    }

    void prune(size_t index)
    {
        BaseType::prune(index, mean(index));
    }

    using BaseType::size;

protected:
    void resetData()
    {
        const size_t cnt = size() * 3;
        for (size_t i = 0; i < cnt; ++i)
        {
            _data[i] = algorithmFPType(0);
        }
    }

private:
    algorithmFPType * _data;
};

template <typename algorithmFPType, CpuType cpu>
static void copyNode(size_t srcNodeIdx, size_t destNodeIdx, const decision_tree::internal::Tree<cpu, algorithmFPType, algorithmFPType> & src,
                     decision_tree::regression::DecisionTreeNode * dest, double *impVals, int *smplCntVals, size_t & destNodeCount, size_t destNodeCapacity,
                     const decision_tree::internal::PruningData<cpu, algorithmFPType> & pruningData)
{
    if (src[srcNodeIdx].isLeaf())
    {
        dest[destNodeIdx].dimension = static_cast<size_t>(-1);
        dest[destNodeIdx].leftIndex = 0;
        dest[destNodeIdx].cutPointOrDependantVariable = src[srcNodeIdx].dependentVariable();
        impVals[destNodeIdx] = src[srcNodeIdx].impurity();
        smplCntVals[destNodeIdx] = src[srcNodeIdx].count();
    }
    else if (pruningData.isPruned(srcNodeIdx))
    {
        dest[destNodeIdx].dimension = static_cast<size_t>(-1);
        dest[destNodeIdx].leftIndex = 0;
        dest[destNodeIdx].cutPointOrDependantVariable = pruningData.dependentVariable(srcNodeIdx);
        impVals[destNodeIdx] = src[srcNodeIdx].impurity();
        smplCntVals[destNodeIdx] = src[srcNodeIdx].count();
    }
    else
    {
        DAAL_ASSERT(destNodeCount + 2 <= destNodeCapacity);
        dest[destNodeIdx].dimension = src[srcNodeIdx].featureIndex();
        const size_t childIndex = destNodeCount;
        dest[destNodeIdx].leftIndex = childIndex;
        dest[destNodeIdx].cutPointOrDependantVariable = src[srcNodeIdx].cutPoint();
        impVals[destNodeIdx] = src[srcNodeIdx].impurity();
        smplCntVals[destNodeIdx] = src[srcNodeIdx].count();
        destNodeCount += 2;
        copyNode(src[srcNodeIdx].leftChildIndex(), childIndex, src, dest, impVals, smplCntVals, destNodeCount, destNodeCapacity, pruningData);
        copyNode(src[srcNodeIdx].rightChildIndex(), childIndex + 1, src, dest, impVals, smplCntVals, destNodeCount, destNodeCapacity, pruningData);
    }
}

template <typename algorithmFPType, CpuType cpu>
services::Status DecisionTreeTrainBatchKernel<algorithmFPType, training::defaultDense, cpu>::
    compute(const NumericTable * x, const NumericTable * y, const NumericTable * px, const NumericTable * py,
            decision_tree::regression::Model * r, const daal::algorithms::Parameter * par)
{
    DAAL_ASSERT(x);
    DAAL_ASSERT(y);
    DAAL_ASSERT(r);
    const decision_tree::regression::Parameter * const parameter = static_cast<const decision_tree::regression::Parameter *>(par);
    DAAL_ASSERT(parameter);

    DAAL_ASSERT(r->impl());
    r->impl()->setNumberOfFeatures(x->getNumberOfColumns());

    Tree<cpu, algorithmFPType, algorithmFPType> tree;
    MSE<algorithmFPType, cpu> splitCriterion;
    tree.train(splitCriterion, *x, *y, 0, parameter->maxTreeDepth, parameter->minObservationsInLeafNodes);
    services::Status status;
    if (parameter->pruning == reducedErrorPruning)
    {
        DAAL_ASSERT(px);
        DAAL_ASSERT(py);
        REPPruningData<algorithmFPType, cpu> repData(tree.nodeCount());
        tree.reducedErrorPruning(*px, *py, repData);

        const size_t nodeCapacity = countNodes(0, tree, repData);
        DecisionTreeTablePtr treeTable(new DecisionTreeTable(nodeCapacity, status));
        services::SharedPtr<data_management::HomogenNumericTable<double> > impTbl(new HomogenNumericTable<double>(1, nodeCapacity, NumericTable::doAllocate));
        services::SharedPtr<data_management::HomogenNumericTable<int> > smplCntTbl(new HomogenNumericTable<int>(1, nodeCapacity, NumericTable::doAllocate));
        DAAL_CHECK_STATUS_VAR(status);
        DecisionTreeNode * const nodes = static_cast<DecisionTreeNode *>(treeTable->getArray());
        double *impVals = impTbl->getArray();
        int *smplCntVals = smplCntTbl->getArray();
        size_t nodeCount = 1;
        copyNode(0, 0, tree, nodes, impVals, smplCntVals, nodeCount, nodeCapacity, repData);
        r->impl()->setTreeTable(treeTable);
        r->impl()->setImpTable(impTbl);
        r->impl()->setNodeSmplCntTable(smplCntTbl);
    }
    else
    {
        const size_t nodeCount = tree.nodeCount();
        DecisionTreeTablePtr treeTable(new DecisionTreeTable(nodeCount, status));
        services::SharedPtr<data_management::HomogenNumericTable<double> > impTbl(new HomogenNumericTable<double>(1, nodeCount, NumericTable::doAllocate));
        services::SharedPtr<data_management::HomogenNumericTable<int> > smplCntTbl(new HomogenNumericTable<int>(1, nodeCount, NumericTable::doAllocate));

        DAAL_CHECK_STATUS_VAR(status);
        DecisionTreeNode * const nodes = static_cast<DecisionTreeNode *>(treeTable->getArray());
        double *impVals = impTbl->getArray();
        int *smplCntVals = smplCntTbl->getArray();

        for (size_t i = 0; i < nodeCount; ++i)
        {
            if (tree[i].isLeaf())
            {
                nodes[i].dimension = static_cast<size_t>(-1);
                nodes[i].leftIndex = 0;
                nodes[i].cutPointOrDependantVariable = tree[i].dependentVariable();
            }
            else
            {
                nodes[i].dimension = tree[i].featureIndex();
                nodes[i].leftIndex = tree[i].leftChildIndex();
                nodes[i].cutPointOrDependantVariable = tree[i].cutPoint();
            }
            impVals[i] = tree[i].impurity();
            smplCntVals[i] = tree[i].count();
        }
        r->impl()->setTreeTable(treeTable);
        r->impl()->setImpTable(impTbl);
        r->impl()->setNodeSmplCntTable(smplCntTbl);
    }
    return status;
}

} // namespace internal
} // namespace training
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
