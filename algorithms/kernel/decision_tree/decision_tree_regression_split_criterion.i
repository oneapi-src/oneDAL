/* file: decision_tree_regression_split_criterion.i */
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

#ifndef __DECISION_TREE_REGRESSION_SPLIT_CRITERION_I__
#define __DECISION_TREE_REGRESSION_SPLIT_CRITERION_I__

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

    MSEDataStatistics(size_t size, const NumericTable * w = nullptr) : _mean(0), _count(0), _mse(0) {}

    MSEDataStatistics(size_t size, const NumericTable & x, const NumericTable & y, const NumericTable * w) : _mean(0), _count(0), _mse(0)
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

    size_t sumWeights(size_t firstIndex, size_t lastIndex, NumericTable * w) const { return (lastIndex - firstIndex); }

    static algorithmFPType subtractMean(algorithmFPType nab, algorithmFPType mab, algorithmFPType na, algorithmFPType nb, algorithmFPType ma)
    {
        const algorithmFPType mb = (nab * mab - na * ma) / nb;
        return mb;
    }

    static algorithmFPType subtractMSE(algorithmFPType vab, algorithmFPType va, algorithmFPType nab, algorithmFPType mab, algorithmFPType na,
                                       algorithmFPType nb, algorithmFPType ma)
    {
        const algorithmFPType delta = subtractMean(nab, mab, na, nb, ma) - ma;
        const algorithmFPType vb    = vab - va - delta * delta * na * nb / nab;
        return vb;
    }

    MSEDataStatistics & operator-=(const MSEDataStatistics & rhs)
    {
        const algorithmFPType newCount = _count - rhs._count;
        const algorithmFPType newMean  = subtractMean(_count, _mean, rhs._count, newCount, rhs._mean);
        _mse                           = subtractMSE(_mse, rhs._mse, _count, _mean, rhs._count, newCount, rhs._mean);
        _count                         = newCount;
        _mean                          = newMean;
        return *this;
    }

    algorithmFPType getBestDependentVariableValue() const { return _mean; }

    void reset(const MSEDataStatistics &)
    {
        _mean  = algorithmFPType(0);
        _count = algorithmFPType(0);
        _mse   = algorithmFPType(0);
    }

    void update(size_t index, algorithmFPType v)
    {
        DAAL_ASSERT(0)
        DAAL_ASSERT(1)
    }

    void update(algorithmFPType v)
    {
        // Welford running method.

        if (++_count == 1)
        {
            _mean = v;
            _mse  = algorithmFPType(0);
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

    size_t operator[](size_t index) const { return 0; }

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
class MSEWeightedDataStatistics
{
public:
    MSEWeightedDataStatistics() : _mean(0), _mse(0), _W(0) {}

    MSEWeightedDataStatistics(size_t size, const NumericTable * w = nullptr) : _mean(0), _mse(0), _W(0) {}

    MSEWeightedDataStatistics(size_t size, const NumericTable & x, const NumericTable & y, const NumericTable * w) : _mean(0), _mse(0), _W(0)
    {
        const size_t nRows = y.getNumberOfRows();
        BlockDescriptor<algorithmFPType> yBD, wBD;
        const_cast<NumericTable &>(y).getBlockOfColumnValues(0, 0, nRows, readOnly, yBD);
        const_cast<NumericTable *>(w)->getBlockOfColumnValues(0, 0, nRows, readOnly, wBD);
        const algorithmFPType * const dy = yBD.getBlockPtr();
        const algorithmFPType * const dw = wBD.getBlockPtr();
        for (size_t i = 0; i < nRows; ++i)
        {
            update(dy[i], dw[i]);
        }
        const_cast<NumericTable &>(y).releaseBlockOfColumnValues(yBD);
        const_cast<NumericTable *>(w)->releaseBlockOfColumnValues(wBD);
    }

    size_t size() const { return 1; }

    size_t sumWeights(size_t firstIndex, size_t lastIndex, NumericTable * w) const
    {
        DAAL_ASSERT(w != nullptr)
        DAAL_ASSERT(firstIndex <= lastIndex)
        DAAL_ASSERT(w->getNumberOfRows() >= lastIndex)
        size_t count = lastIndex - firstIndex;
        BlockDescriptor<algorithmFPType> wBD;
        const_cast<NumericTable *>(w)->getBlockOfColumnValues(0, firstIndex, count, readOnly, wBD);
        const algorithmFPType * const dw = wBD.getBlockPtr();
        algorithmFPType sum              = 0.0;
        for (size_t i = 0; i < count; i++)
        {
            sum += dw[i];
        }
        const_cast<NumericTable *>(w)->releaseBlockOfColumnValues(wBD);
        return sum;
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
        const algorithmFPType vb    = vab - va - delta * delta * na * nb / nab;
        return vb;
    }

    MSEWeightedDataStatistics & operator-=(const MSEWeightedDataStatistics & rhs)
    {
        const algorithmFPType newW    = _W - rhs._W;
        const algorithmFPType newMean = subtractMean(_W, _mean, rhs._W, newW, rhs._mean);
        _mse                          = subtractMSE(_mse, rhs._mse, _W, _mean, rhs._W, newW, rhs._mean);
        _W                            = newW;
        _mean                         = newMean;
        return *this;
    }

    algorithmFPType getBestDependentVariableValue() const { return _mean; }

    void reset(const MSEWeightedDataStatistics &)
    {
        _mean = algorithmFPType(0);
        _mse  = algorithmFPType(0);
        _W    = algorithmFPType(0);
    }

    void update(size_t index, algorithmFPType v) { DAAL_ASSERT(0) }

    void update(algorithmFPType v, algorithmFPType weight = 0)
    {
        if (weight == 0) return;

        // Welford running method.
        _W += weight;
        if (_W == weight)
        {
            _mean = v;
            _mse  = algorithmFPType(0);
        }
        else
        {
            const algorithmFPType delta = v - _mean;
            _mean += weight * delta / _W;
            _mse += weight * delta * (v - _mean);
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

    algorithmFPType mse() const { return _mse; }

    algorithmFPType operator[](size_t index) const { return _W; }

    void swap(MSEWeightedDataStatistics & value)
    {
        services::internal::swap<cpu>(_mean, value._mean);
        services::internal::swap<cpu>(_mse, value._mse);
        services::internal::swap<cpu>(_W, value._W);
    }

private:
    algorithmFPType _mean;
    algorithmFPType _mse;
    algorithmFPType _W;
};

template <typename algorithmFPType, CpuType cpu>
struct MSE
{
    typedef MSEDataStatistics<algorithmFPType, cpu> DataStatistics;
    typedef algorithmFPType ValueType;
    typedef algorithmFPType DependentVariableType;

    template <typename RandomIterator>
    ValueType operator()(RandomIterator first, RandomIterator last, RandomIterator current, RandomIterator next, DataStatistics & dataStatistics,
                         const DataStatistics & totalDataStatistics, data_management::features::FeatureType featureType, size_t leftCount,
                         size_t rightCount, size_t totalCount)
    {
        const ValueType leftMSE  = dataStatistics.mse();
        const ValueType rightMSE = DataStatistics::subtractMSE(totalDataStatistics.mse(), leftMSE, totalCount, totalDataStatistics.mean(), leftCount,
                                                               rightCount, dataStatistics.mean());
        return leftMSE + rightMSE;
    }

    ValueType operator()(const DataStatistics & totalDataStatistics, size_t totalCount) { return totalDataStatistics.mse() / totalCount; }
};

template <typename algorithmFPType, CpuType cpu>
struct MSEWeighted
{
    typedef MSEWeightedDataStatistics<algorithmFPType, cpu> DataStatistics;
    typedef algorithmFPType ValueType;
    typedef algorithmFPType DependentVariableType;

    template <typename RandomIterator>
    ValueType operator()(RandomIterator first, RandomIterator last, RandomIterator current, RandomIterator next, DataStatistics & dataStatistics,
                         const DataStatistics & totalDataStatistics, data_management::features::FeatureType featureType, ValueType leftWeight,
                         ValueType rightWeight, ValueType totalWeight)
    {
        const ValueType leftMSE  = dataStatistics.mse();
        const ValueType rightMSE = DataStatistics::subtractMSE(totalDataStatistics.mse(), leftMSE, totalWeight, totalDataStatistics.mean(),
                                                               leftWeight, rightWeight, dataStatistics.mean());
        return leftMSE + rightMSE;
    }

    ValueType operator()(const DataStatistics & totalDataStatistics, size_t totalWeight) { return totalDataStatistics.mse() / totalWeight; }
};

} // namespace internal
} // namespace training
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
