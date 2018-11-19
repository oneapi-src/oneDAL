/* file: indices.h */
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

#ifndef __DATA_MANAGEMENT__FEATURES_INDICES_H__
#define __DATA_MANAGEMENT__FEATURES_INDICES_H__

#include <limits>

#include "services/daal_string.h"
#include "services/buffer_view.h"
#include "services/daal_shared_ptr.h"

namespace daal
{
namespace data_management
{
namespace features
{
namespace interface1
{

/**
 * Type that represents index of the feature in the data set
 */
typedef size_t FeatureIndex;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__FEATUREINDICESIFACE"></a>
 * \brief Static class that contains auxiliary methods for FeatureIndex
 */
class FeatureIndexTraits
{
public:
    /**
     * Returns the index that describes invalid value of FeatureIndex
     */
    static FeatureIndex invalid()
    {
        return (std::numeric_limits<FeatureIndex>::max)();
    }

    /**
     * Returns maximal available value of FeatureIndex
     */
    static FeatureIndex maxIndex()
    {
        return (std::numeric_limits<FeatureIndex>::max)() - 1;
    }

private:
    FeatureIndexTraits();
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__FEATUREINDICESIFACE"></a>
 * \brief Abstract class that defines interface for feature indices collection
 */
class FeatureIndicesIface
{
public:
    virtual ~FeatureIndicesIface() { }

    /**
     * Returns the number of feature indices in the collection
     */
    virtual size_t size() const = 0;

    /**
     * Checks if collection represents continuous range of indices. If method returns True,
     * the methods getFirst() and getLast() return lower and upper bounds of the range
     * \return True if the feature indices collection contains continuous range of indices
     */
    virtual bool isPlainRange() const = 0;

    /**
     * Checks if the raw array of feature indices is available
     * \return True if the raw array of indices is available
     */
    virtual bool areRawFeatureIndicesAvailable() const = 0;

    /**
     * Gets the first index in the collection
     * \return The very first index in the collection. If collection is
     *         empty, method returns FeatureIndexTraits::invalid() value
     */
    virtual FeatureIndex getFirst() const = 0;

    /**
     * Gets the last index in the collection
     * \return The last index in the collection. If collection is
     *         empty, method returns FeatureIndexTraits::invalid() value
     */
    virtual FeatureIndex getLast() const = 0;

    /**
     * Gets the raw array that stores all feature indices available in the collection
     * \return The buffer view object that contains continuous sequence of features
     */
    virtual services::BufferView<FeatureIndex> getRawFeatureIndices() = 0;
};
typedef services::SharedPtr<FeatureIndicesIface> FeatureIndicesIfacePtr;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__FEATUREINDICES"></a>
 * \brief Base class that represents the collection of feature indices,
 *        intended for inheritance from the user side
 */
class FeatureIndices : public Base, public FeatureIndicesIface { };
typedef services::SharedPtr<FeatureIndices> FeatureIndicesPtr;

} // namespace interface1

using interface1::FeatureIndex;
using interface1::FeatureIndexTraits;
using interface1::FeatureIndicesIface;
using interface1::FeatureIndicesIfacePtr;
using interface1::FeatureIndices;
using interface1::FeatureIndicesPtr;

} // namespace features
} // namespace data_management
} // namespace daal

#endif
