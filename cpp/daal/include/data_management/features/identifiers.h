/* file: identifiers.h */
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

#ifndef __DATA_MANAGEMENT_FEATURES_IDENTIFIERS_H__
#define __DATA_MANAGEMENT_FEATURES_IDENTIFIERS_H__

#include "services/daal_string.h"
#include "services/daal_shared_ptr.h"

#include "data_management/features/defines.h"
#include "data_management/features/indices.h"

namespace daal
{
namespace data_management
{
namespace features
{
namespace interface1
{
/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__FEATUREINDICESIFACE"></a>
 * \brief Abstract class that defines interface for mapping feature id to feature index
 */
class FeatureIdMappingIface
{
public:
    virtual ~FeatureIdMappingIface() {}

    /**
     * Gets the number of features
     * \return The number of features
     */
    virtual size_t getNumberOfFeatures() const = 0;

    /**
     * Checks if keys for corresponding feature indices are available
     * \return True if keys are available
     */
    virtual bool areKeysAvailable() const = 0;

    /**
     * Returns feature index for corresponding key
     * \param[in]  key   The string identifier that represents a key
     * \return     Feature index for specified key. If key is not
     *             found method returns FeatureIndexTraits::invalid()
     */
    virtual FeatureIndex getIndexByKey(const services::String & key) const = 0;
};
typedef services::SharedPtr<FeatureIdMappingIface> FeatureIdMappingIfacePtr;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__FEATUREIDMAPPING"></a>
 * \brief Base class that partially implements feature mapping interface,
 *        intended for inheritance form the user side
 */
class FeatureIdMapping : public Base, public FeatureIdMappingIface
{};
typedef services::SharedPtr<FeatureIdMapping> FeatureIdMappingPtr;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__FEATUREIDIFACE"></a>
 * \brief Abstract feature id interface
 */
class FeatureIdIface
{
public:
    virtual ~FeatureIdIface() {}

    /**
     * Maps feature id to the respective feature index using given feature mapping
     * \param[in]  mapping  The feature mapping
     * \param[out] status   The status object
     * \return Feature index for the given feature id. If key is not found
     *         method returns FeatureIndexTraits::invalid()
     */
    virtual FeatureIndex mapToIndex(const FeatureIdMappingIface & mapping, services::Status * status = NULL) = 0;
};
typedef services::SharedPtr<FeatureIdIface> FeatureIdIfacePtr;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__FEATUREID"></a>
 * \brief Base class that partially implements abstract feature id,
 *        intended for inheritance form user side
 */
class FeatureId : public Base, public FeatureIdIface
{};
typedef services::SharedPtr<FeatureId> FeatureIdPtr;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__FEATUREIDCOLLECTIONIFACE"></a>
 * \brief Abstract class that represents collection of feature ids
 */
class FeatureIdCollectionIface
{
public:
    virtual ~FeatureIdCollectionIface() {}

    /**
     * Maps collection of feature ids to the respective collection
     * of feature indices using given feature mapping
     * \param[in]  mapping  The feature mapping
     * \param[out] status   The status
     * \return Shared pointer to the collection of feature indices
     */
    virtual FeatureIndicesIfacePtr mapToFeatureIndices(const FeatureIdMappingIface & mapping, services::Status * status = NULL) = 0;
};
typedef services::SharedPtr<FeatureIdCollectionIface> FeatureIdCollectionIfacePtr;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__FEATUREIDCOLLECTIONIFACE"></a>
 * \brief Base class that partially implements abstract feature id collection,
 *        intended for inheritance form user side
 */
class FeatureIdCollection : public Base, public FeatureIdCollectionIface
{};
typedef services::SharedPtr<FeatureIdCollection> FeatureIdCollectionPtr;

} // namespace interface1

using interface1::FeatureIdMappingIface;
using interface1::FeatureIdMappingIfacePtr;
using interface1::FeatureIdMapping;
using interface1::FeatureIdMappingPtr;

using interface1::FeatureIdIface;
using interface1::FeatureIdIfacePtr;
using interface1::FeatureId;
using interface1::FeatureIdPtr;

typedef interface1::FeatureIdCollectionIface FeatureIdCollectionIface;
typedef services::SharedPtr<interface1::FeatureIdCollectionIface> FeatureIdCollectionIfacePtr;
using interface1::FeatureIdCollection;
using interface1::FeatureIdCollectionPtr;

} // namespace features
} // namespace data_management
} // namespace daal

#endif
