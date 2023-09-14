/* file: decision_tree_classification_split_criterion.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#ifndef __DECISION_TREE_CLASSIFICATION_SPLIT_CRITERION_I__
#define __DECISION_TREE_CLASSIFICATION_SPLIT_CRITERION_I__

#include "services/daal_defines.h"
#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "src/services/service_utils.h"
#include "data_management/data/numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services::internal;
using namespace decision_tree::internal;

template <CpuType cpu>
class ClassCounters
{
public:
    ClassCounters() : _size(0), _counters(nullptr) {}

    ClassCounters(const ClassCounters & value) : _size(value._size), _counters(value._size ? daal_alloc<size_t>(value._size) : nullptr)
    {
        int result = daal::services::internal::daal_memcpy_s(_counters, _size * sizeof(size_t), value._counters, value._size * sizeof(size_t));
        _status |= (result) ? services::Status(services::ErrorMemoryCopyFailedInternal) : _status;
    }

    ClassCounters(size_t size, const NumericTable * w = nullptr) : _size(size), _counters(size ? daal_alloc<size_t>(size) : nullptr) { reset(); }

    ClassCounters(size_t size, const NumericTable & x, const NumericTable & y, const NumericTable * w)
        : _size(size), _counters(size ? daal_alloc<size_t>(size) : nullptr)
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
        _counters = nullptr;
    }

    size_t sumWeights(size_t firstIndex, size_t lastIndex, NumericTable * w) { return (lastIndex - firstIndex); }

    ClassCounters & operator=(const ClassCounters & rhs)
    {
        if (rhs._size != _size)
        {
            daal_free(_counters);
            _counters = daal_alloc<size_t>(rhs._size);
            _size     = rhs._size;
        }
        int result = daal::services::internal::daal_memcpy_s(_counters, _size * sizeof(size_t), rhs._counters, rhs._size * sizeof(size_t));
        _status |= (result) ? services::Status(services::ErrorMemoryCopyFailedInternal) : _status;
        return *this;
    }

    ClassCounters & operator-=(const ClassCounters & rhs)
    {
        DAAL_ASSERT(_size == rhs._size);
        for (size_t i = 0; i < _size; ++i)
        {
            DAAL_ASSERT(_counters[i] >= rhs._counters[i]);
            _counters[i] -= rhs._counters[i];
        }
        return *this;
    }

    size_t operator[](size_t index) const
    {
        DAAL_ASSERT(index < _size);
        return _counters[index];
    }

    void swap(ClassCounters & value)
    {
        services::internal::swap<cpu>(_counters, value._counters);
        services::internal::swap<cpu>(_size, value._size);
    }

    size_t getBestDependentVariableValue() const { return maxElement<cpu>(_counters, &_counters[_size]) - _counters; }

    void reset(const ClassCounters & value)
    {
        if (_size != value._size)
        {
            _size                 = value._size;
            size_t * saveCounters = _counters;
            _counters             = _size ? daal_alloc<size_t>(_size) : nullptr;
            daal_free(saveCounters);
            saveCounters = nullptr;
        }
        reset();
    }

    void reset()
    {
        for (size_t i = 0; i < _size; ++i)
        {
            _counters[i] = 0;
        }
    }

    void update(size_t index, float weight = 1.0)
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
                if (numberOfClasses >= 2)
                {
                    break;
                }
                onlyClass = i;
            }
        }
        return (numberOfClasses == 1);
    }

    size_t size() const { return _size; }

    void putProbabilities(double * probs, size_t numProbs) const
    {
        DAAL_ASSERT(probs);
        DAAL_ASSERT(_counters);
        DAAL_ASSERT(numProbs == _size);

        size_t total = 0;
        for (size_t i = 0; i < _size; ++i)
        {
            total += _counters[i];
        }

        if (total != 0)
        {
            for (size_t i = 0; i < _size; ++i)
            {
                probs[i] = static_cast<double>(_counters[i]) / static_cast<double>(total);
            }
        }
        else
        {
            for (size_t i = 0; i < _size; ++i)
            {
                probs[i] = 0.0;
            }
        }
    }

    bool ok() const { return _status.ok(); }
    services::Status getLastStatus() const { return _status; }

protected:
    services::Status _status;

private:
    size_t _size;
    size_t * _counters;
};

template <typename algorithmFPType, CpuType cpu>
class ClassWeightsCounters
{
public:
    ClassWeightsCounters() : _size(0), _counters(nullptr) {}

    ClassWeightsCounters(const ClassWeightsCounters & value)
        : _size(value._size), _counters(value._size ? daal_alloc<algorithmFPType>(value._size) : nullptr)
    {
        int result = daal::services::internal::daal_memcpy_s(_counters, _size * sizeof(algorithmFPType), value._counters,
                                                             value._size * sizeof(algorithmFPType));
        _status |= (result) ? services::Status(services::ErrorMemoryCopyFailedInternal) : _status;
    }

    ClassWeightsCounters(size_t size, const NumericTable * w) : _size(size), _counters(size ? daal_alloc<algorithmFPType>(size) : nullptr)
    {
        reset();
    }

    ClassWeightsCounters(size_t size, const NumericTable & x, const NumericTable & y, const NumericTable * w)
        : _size(size), _counters(size ? daal_alloc<algorithmFPType>(size) : nullptr)
    {
        DAAL_ASSERT(w != nullptr)
        if (size)
        {
            reset();
            const size_t yRowCount = y.getNumberOfRows();
            BlockDescriptor<int> yBD;
            BlockDescriptor<algorithmFPType> wBD;
            const_cast<NumericTable &>(y).getBlockOfColumnValues(0, 0, yRowCount, readOnly, yBD);
            const_cast<NumericTable *>(w)->getBlockOfColumnValues(0, 0, yRowCount, readOnly, wBD);
            const int * const dy             = yBD.getBlockPtr();
            const algorithmFPType * const dw = wBD.getBlockPtr();
            for (size_t i = 0; i < yRowCount; ++i)
            {
                update(static_cast<size_t>(dy[i]), dw[i]);
            }
            const_cast<NumericTable &>(y).releaseBlockOfColumnValues(yBD);
            const_cast<NumericTable *>(w)->releaseBlockOfColumnValues(wBD);
        }
    }

    ~ClassWeightsCounters()
    {
        daal_free(_counters);
        _counters = nullptr;
    }

    algorithmFPType sumWeights(size_t firstIndex, size_t lastIndex, NumericTable * w)
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

    ClassWeightsCounters & operator=(const ClassWeightsCounters & rhs)
    {
        if (rhs._size != _size)
        {
            daal_free(_counters);
            _counters = daal_alloc<algorithmFPType>(rhs._size);
            _size     = rhs._size;
        }
        int result =
            daal::services::internal::daal_memcpy_s(_counters, _size * sizeof(algorithmFPType), rhs._counters, rhs._size * sizeof(algorithmFPType));
        _status |= (result) ? services::Status(services::ErrorMemoryCopyFailedInternal) : _status;
        return *this;
    }

    ClassWeightsCounters & operator-=(const ClassWeightsCounters & rhs)
    {
        DAAL_ASSERT(_size == rhs._size);
        for (size_t i = 0; i < _size; ++i)
        {
            DAAL_ASSERT((_counters[i] + 0.001) > rhs._counters[i]);
            _counters[i] -= rhs._counters[i];
        }
        return *this;
    }

    algorithmFPType operator[](size_t index) const
    {
        DAAL_ASSERT(index < _size);
        return _counters[index];
    }

    void swap(ClassWeightsCounters & value)
    {
        services::internal::swap<cpu>(_counters, value._counters);
        services::internal::swap<cpu>(_size, value._size);
    }

    size_t getBestDependentVariableValue() const { return maxElement<cpu>(_counters, &_counters[_size]) - _counters; }

    void reset(const ClassWeightsCounters & value)
    {
        if (_size != value._size)
        {
            _size                          = value._size;
            algorithmFPType * saveCounters = _counters;
            _counters                      = _size ? daal_alloc<algorithmFPType>(_size) : nullptr;
            daal_free(saveCounters);
            saveCounters = nullptr;
        }
        reset();
    }

    void reset()
    {
        for (size_t i = 0; i < _size; ++i)
        {
            _counters[i] = 0;
        }
    }

    void update(size_t index, algorithmFPType weight = 0.0)
    {
        DAAL_ASSERT(index < _size);
        _counters[index] += weight;
    }

    bool isPure(size_t & onlyClass) const
    {
        size_t numberOfClasses = 0;
        for (size_t i = 0; i < _size; ++i)
        {
            if (_counters[i])
            {
                ++numberOfClasses;
                if (numberOfClasses >= 2)
                {
                    break;
                }
                onlyClass = i;
            }
        }
        return (numberOfClasses == 1);
    }

    size_t size() const { return _size; }

    void putProbabilities(double * probs, size_t numProbs) const
    {
        DAAL_ASSERT(probs);
        DAAL_ASSERT(_counters);
        DAAL_ASSERT(numProbs == _size);

        algorithmFPType total = 0.0;
        for (size_t i = 0; i < _size; ++i)
        {
            total += _counters[i];
        }

        if (total != 0.0)
        {
            for (size_t i = 0; i < _size; ++i)
            {
                probs[i] = _counters[i] / total;
            }
        }
        else
        {
            for (size_t i = 0; i < _size; ++i)
            {
                probs[i] = 0.0;
            }
        }
    }

    bool ok() const { return _status.ok(); }
    services::Status getLastStatus() const { return _status; }

protected:
    services::Status _status;

private:
    size_t _size;
    algorithmFPType * _counters;
};

template <typename algorithmFPType, CpuType cpu>
struct Gini
{
    typedef ClassCounters<cpu> DataStatistics;
    typedef algorithmFPType ValueType;
    typedef size_t DependentVariableType;

    template <typename RandomIterator>
    ValueType operator()(RandomIterator first, RandomIterator last, RandomIterator current, RandomIterator next, DataStatistics & dataStatistics,
                         const DataStatistics & totalDataStatistics, data_management::features::FeatureType featureType, size_t leftCount,
                         size_t rightCount, size_t totalCount)
    {
        const ValueType leftProbability  = leftCount * static_cast<ValueType>(1) / totalCount;
        const ValueType rightProbability = rightCount * static_cast<ValueType>(1) / totalCount;
        ValueType leftGini               = 1;
        ValueType rightGini              = 1;
        const size_t size                = dataStatistics.size();
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

    ValueType operator()(const DataStatistics & totalDataStatistics, size_t totalCount)
    {
        ValueType gini    = 1;
        const size_t size = totalDataStatistics.size();
        for (size_t i = 0; i < size; ++i)
        {
            const ValueType leftP = totalDataStatistics[i] * static_cast<ValueType>(1) / totalCount;
            gini -= leftP * leftP;
        }
        return gini;
    }
};

template <typename algorithmFPType, CpuType cpu>
struct GiniWeighted
{
    typedef ClassWeightsCounters<algorithmFPType, cpu> DataStatistics;
    typedef algorithmFPType ValueType;
    typedef size_t DependentVariableType;

    template <typename RandomIterator>
    ValueType operator()(RandomIterator first, RandomIterator last, RandomIterator current, RandomIterator next,
                         const DataStatistics & dataStatistics, const DataStatistics & totalDataStatistics,
                         data_management::features::FeatureType featureType, ValueType leftWeight, ValueType rightWeight, ValueType totalWeight)
    {
        const ValueType invTotalWeight   = static_cast<ValueType>(1) / totalWeight;
        const ValueType invRightWeight   = static_cast<ValueType>(1) / rightWeight;
        const ValueType invLeftWeight    = static_cast<ValueType>(1) / leftWeight;
        const ValueType leftProbability  = leftWeight * invTotalWeight;
        const ValueType rightProbability = rightWeight * invTotalWeight;
        ValueType leftGini               = leftWeight * leftWeight;
        ValueType rightGini              = rightWeight * rightWeight;
        const size_t size                = dataStatistics.size();
        DAAL_ASSERT(size == totalDataStatistics.size());
        for (size_t i = 0; i < size; ++i)
        {
            const ValueType leftP = dataStatistics[i];
            leftGini -= leftP * leftP;
            const ValueType rightP = (totalDataStatistics[i] - dataStatistics[i]);
            rightGini -= rightP * rightP;
        }
        return leftProbability * leftGini * invLeftWeight * invLeftWeight + rightProbability * rightGini * invRightWeight * invRightWeight;
    }

    ValueType operator()(const DataStatistics & totalDataStatistics, algorithmFPType totalWeight)
    {
        const ValueType sqTotalWeight = totalWeight * totalWeight;
        ValueType gini                = sqTotalWeight;
        const size_t size             = totalDataStatistics.size();
        for (size_t i = 0; i < size; ++i)
        {
            const ValueType leftP = totalDataStatistics[i];
            gini -= leftP * leftP;
        }
        return gini / sqTotalWeight;
    }
};

template <typename algorithmFPType, CpuType cpu>
struct InfoGain
{
    typedef ClassCounters<cpu> DataStatistics;
    typedef algorithmFPType ValueType;
    typedef size_t DependentVariableType;

    InfoGain() : _tempSize(0), _tempP(nullptr), _tempLg(nullptr) {}

    InfoGain(const InfoGain &) : _tempSize(0), _tempP(nullptr), _tempLg(nullptr) {}

    InfoGain & operator=(const InfoGain &) { return *this; }

    ~InfoGain() { deallocateTempData(); }

    template <typename RandomIterator>
    ValueType operator()(RandomIterator first, RandomIterator last, RandomIterator current, RandomIterator next,
                         const DataStatistics & dataStatistics, const DataStatistics & totalDataStatistics,
                         data_management::features::FeatureType featureType, size_t leftCount, size_t rightCount, size_t totalCount)
    {
        typedef MathInst<algorithmFPType, cpu> MathType;

        const ValueType leftProbability  = leftCount * static_cast<ValueType>(1) / totalCount;
        const ValueType rightProbability = rightCount * static_cast<ValueType>(1) / totalCount;
        ValueType leftInfo               = 0;
        ValueType rightInfo              = 0;
        const size_t size                = dataStatistics.size();
        DAAL_ASSERT(size == totalDataStatistics.size());
        if (allocateTempData(size * 2))
        {
            for (size_t i = 0; i < size; ++i)
            {
                const ValueType leftP  = dataStatistics[i] * static_cast<ValueType>(1) / leftCount;
                _tempP[i]              = (leftP != 0) ? leftP : 1;
                const ValueType rightP = (totalDataStatistics[i] - dataStatistics[i]) * static_cast<ValueType>(1) / rightCount;
                _tempP[size + i]       = (rightP != 0) ? rightP : 1;
            }
            MathType::vLog(size * 2, _tempP, _tempLg);
            for (size_t i = 0; i < size; ++i)
            {
                leftInfo -= _tempP[i] * _tempLg[i];
                rightInfo -= _tempP[size + i] * _tempLg[size + i];
            }
        }
        else
        {
            for (size_t i = 0; i < size; ++i)
            {
                const ValueType leftP = dataStatistics[i] * static_cast<ValueType>(1) / leftCount;
                leftInfo -= (leftP != 0) ? leftP * MathType::sLog(leftP) : 0;
                const ValueType rightP = (totalDataStatistics[i] - dataStatistics[i]) * static_cast<ValueType>(1) / rightCount;
                rightInfo -= (rightP != 0) ? rightP * MathType::sLog(rightP) : 0;
            }
        }
        return leftProbability * leftInfo + rightProbability * rightInfo;
    }

    ValueType operator()(const DataStatistics & totalDataStatistics, size_t totalCount)
    {
        typedef MathInst<algorithmFPType, cpu> MathType;

        ValueType info    = 0;
        const size_t size = totalDataStatistics.size();

        for (size_t i = 0; i < size; ++i)
        {
            const ValueType leftP = totalDataStatistics[i] * static_cast<ValueType>(1) / totalCount;
            info -= (leftP != 0) ? leftP * MathType::sLog(leftP) : 0;
        }

        return info;
    }

protected:
    bool allocateTempData(size_t size)
    {
        DAAL_ASSERT((_tempSize == 0) == (_tempP == nullptr));
        DAAL_ASSERT((_tempSize == 0) == (_tempLg == nullptr));

        if (_tempSize >= size)
        {
            return true;
        }

        deallocateTempData();
        _tempSize = size;
        _tempP    = services::internal::service_scalable_calloc<ValueType, cpu>(size);
        _tempLg   = services::internal::service_scalable_calloc<ValueType, cpu>(size);
        return _tempP != nullptr && _tempLg != nullptr;
    }

    void deallocateTempData()
    {
        services::internal::service_scalable_free<ValueType, cpu>(_tempP);
        services::internal::service_scalable_free<ValueType, cpu>(_tempLg);
        _tempP  = nullptr;
        _tempLg = nullptr;
    }

private:
    size_t _tempSize;
    ValueType * _tempP;
    ValueType * _tempLg;
};

template <typename algorithmFPType, CpuType cpu>
struct InfoGainWeighted
{
    typedef ClassWeightsCounters<algorithmFPType, cpu> DataStatistics;
    typedef algorithmFPType ValueType;
    typedef size_t DependentVariableType;

    InfoGainWeighted() : _tempSize(0), _tempP(nullptr), _tempLg(nullptr) {}

    InfoGainWeighted(const InfoGainWeighted &) : _tempSize(0), _tempP(nullptr), _tempLg(nullptr) {}

    InfoGainWeighted & operator=(const InfoGainWeighted &) { return *this; }

    ~InfoGainWeighted() { deallocateTempData(); }

    template <typename RandomIterator>
    ValueType operator()(RandomIterator first, RandomIterator last, RandomIterator current, RandomIterator next, DataStatistics & dataStatistics,
                         const DataStatistics & totalDataStatistics, data_management::features::FeatureType featureType, ValueType leftWeight,
                         ValueType rightWeight, ValueType totalWeight)
    {
        typedef MathInst<algorithmFPType, cpu> MathType;

        const ValueType leftProbability  = leftWeight * static_cast<ValueType>(1) / totalWeight;
        const ValueType rightProbability = rightWeight * static_cast<ValueType>(1) / totalWeight;
        ValueType leftInfo               = 0;
        ValueType rightInfo              = 0;
        const size_t size                = dataStatistics.size();
        DAAL_ASSERT(size == totalDataStatistics.size());
        if (allocateTempData(size * 2))
        {
            for (size_t i = 0; i < size; ++i)
            {
                const ValueType leftP  = dataStatistics[i] * static_cast<ValueType>(1) / leftWeight;
                _tempP[i]              = (leftP != 0) ? leftP : 1;
                const ValueType rightP = (totalDataStatistics[i] - dataStatistics[i]) * static_cast<ValueType>(1) / rightWeight;
                _tempP[size + i]       = (rightP != 0) ? rightP : 1;
            }
            MathType::vLog(size * 2, _tempP, _tempLg);
            for (size_t i = 0; i < size; ++i)
            {
                leftInfo -= _tempP[i] * _tempLg[i];
                rightInfo -= _tempP[size + i] * _tempLg[size + i];
            }
        }
        else
        {
            for (size_t i = 0; i < size; ++i)
            {
                const ValueType leftP = dataStatistics[i] * static_cast<ValueType>(1) / leftWeight;
                leftInfo -= (leftP != 0) ? leftP * MathType::sLog(leftP) : 0;
                const ValueType rightP = (totalDataStatistics[i] - dataStatistics[i]) * static_cast<ValueType>(1) / rightWeight;
                rightInfo -= (rightP != 0) ? rightP * MathType::sLog(rightP) : 0;
            }
        }
        return leftProbability * leftInfo + rightProbability * rightInfo;
    }

    ValueType operator()(const DataStatistics & totalDataStatistics, algorithmFPType totalWeight)
    {
        typedef MathInst<algorithmFPType, cpu> MathType;

        ValueType info    = 0;
        const size_t size = totalDataStatistics.size();

        for (size_t i = 0; i < size; ++i)
        {
            const ValueType leftP = totalDataStatistics[i] * static_cast<ValueType>(1) / totalWeight;
            info -= (leftP != 0) ? leftP * MathType::sLog(leftP) : 0;
        }

        return info;
    }

protected:
    bool allocateTempData(size_t size)
    {
        DAAL_ASSERT((_tempSize == 0) == (_tempP == nullptr));
        DAAL_ASSERT((_tempSize == 0) == (_tempLg == nullptr));

        if (_tempSize >= size)
        {
            return true;
        }

        deallocateTempData();
        _tempSize = size;
        _tempP    = services::internal::service_scalable_calloc<ValueType, cpu>(size);
        _tempLg   = services::internal::service_scalable_calloc<ValueType, cpu>(size);
        return _tempP != nullptr && _tempLg != nullptr;
    }

    void deallocateTempData()
    {
        services::internal::service_scalable_free<ValueType, cpu>(_tempP);
        services::internal::service_scalable_free<ValueType, cpu>(_tempLg);
        _tempP  = nullptr;
        _tempLg = nullptr;
    }

private:
    size_t _tempSize;
    ValueType * _tempP;
    ValueType * _tempLg;
};

} // namespace internal
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
