/* file: indices_impl.h */
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

#ifndef __DATA_MANAGEMENT_FEATURES_INTERNAL_INDICES_IMPL_H__
#define __DATA_MANAGEMENT_FEATURES_INTERNAL_INDICES_IMPL_H__

#include <map>
#include <string>

#include "services/collection.h"
#include "services/internal/utilities.h"
#include "services/internal/error_handling_helpers.h"

#include "data_management/features/indices.h"

namespace daal
{
namespace data_management
{
namespace features
{
namespace internal
{

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__FEATUREINDICESLIST"></a>
 * \brief Implementation of FeatureIndices to store a list of feature indices
 */
class FeatureIndicesList : public FeatureIndices
{
public:
    static services::SharedPtr<FeatureIndicesList> create(services::Status *status = NULL)
    {
        return services::internal::wrapSharedAndTryThrow<FeatureIndicesList>(new FeatureIndicesList(), status);
    }

    virtual size_t size() const DAAL_C11_OVERRIDE
    {
        return _indices.size();
    }

    virtual bool isPlainRange() const DAAL_C11_OVERRIDE
    {
        return false;
    }

    virtual bool areRawFeatureIndicesAvailable() const DAAL_C11_OVERRIDE
    {
        return true;
    }

    virtual FeatureIndex getFirst() const DAAL_C11_OVERRIDE
    {
        if (!size()) { return FeatureIndexTraits::invalid(); }
        return _indices[0];
    }

    virtual FeatureIndex getLast() const DAAL_C11_OVERRIDE
    {
        if (!size()) { return FeatureIndexTraits::invalid(); }
        return _indices[_indices.size() - 1];
    }

    virtual services::BufferView<FeatureIndex> getRawFeatureIndices() DAAL_C11_OVERRIDE
    {
        return services::BufferView<FeatureIndex>(_indices.data(), _indices.size());
    }

    services::Status add(FeatureIndex index)
    {
        if (index > FeatureIndexTraits::maxIndex() || index == FeatureIndexTraits::invalid())
        {
            return services::throwIfPossible(services::ErrorIncorrectDataRange);
        }

        if ( !_indices.safe_push_back(index) )
        {
            return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
        }

        return services::Status();
    }

private:
    FeatureIndicesList() { }

    services::Collection<FeatureIndex> _indices;
};
typedef services::SharedPtr<FeatureIndicesList> FeatureIndicesListPtr;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__FEATUREINDICESRANGE"></a>
 * \brief Implementation of FeatureIndices to store a range of feature indices
 */
class FeatureIndicesRange : public FeatureIndices
{
public:
    static services::SharedPtr<FeatureIndicesRange> create(FeatureIndex begin, FeatureIndex end,
                                                           services::Status *status = NULL)
    {
        if (begin == FeatureIndexTraits::invalid() ||
            end == FeatureIndexTraits::invalid())
        {
            services::internal::tryAssignStatusAndThrow(status, services::ErrorIncorrectIndex);
            return services::SharedPtr<FeatureIndicesRange>();
        }
        return services::internal::wrapSharedAndTryThrow<FeatureIndicesRange>(
            new FeatureIndicesRange(begin, end), status);
    }

    virtual size_t size() const DAAL_C11_OVERRIDE
    {
        return services::internal::maxValue(_begin, _end) -
               services::internal::minValue(_begin, _end) + 1;
    }

    virtual bool isPlainRange() const DAAL_C11_OVERRIDE
    {
        return true;
    }

    virtual bool areRawFeatureIndicesAvailable() const DAAL_C11_OVERRIDE
    {
        return false;
    }

    virtual FeatureIndex getFirst() const DAAL_C11_OVERRIDE
    {
        return _begin;
    }

    virtual FeatureIndex getLast() const DAAL_C11_OVERRIDE
    {
        return _end;
    }

    virtual services::BufferView<FeatureIndex> getRawFeatureIndices() DAAL_C11_OVERRIDE
    {
        return services::BufferView<FeatureIndex>();
    }

private:
    explicit FeatureIndicesRange(FeatureIndex begin, FeatureIndex end) :
        _begin(begin),
        _end(end) { }

    FeatureIndex _begin;
    FeatureIndex _end;
};
typedef services::SharedPtr<FeatureIndicesList> FeatureIndicesListPtr;

} // namespace internal
} // namespace features
} // namespace data_management
} // namespace daal

#endif
