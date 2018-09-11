/* file: modifier.h */
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

#ifndef __DATA_SOURCE_MODIFIERS_MODIFIER_H__
#define __DATA_SOURCE_MODIFIERS_MODIFIER_H__

#include "services/buffer_view.h"
#include "data_management/data_source/data_source_dictionary.h"

namespace daal
{
namespace data_management
{
/**
 * \brief Contains modifiers components for different Data Sources
 */
namespace modifiers
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @defgroup data_source_modifiers Modifiers
 * \brief Defines special components which can be used to modify
 *        data during the loading through the data source components
 * @ingroup data_sources
 * @{
 */

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CONFIGIFACE"></a>
 * \brief Abstract class that defines interface of modifier configuration
 */
class ConfigIface
{
public:
    virtual ~ConfigIface() { }

    /**
     * Gets the number of input features
     * \return The number of input features
     */
    virtual size_t getNumberOfInputFeatures() const = 0;

    /**
     * Sets the number of output features
     * \param[in] numberOfOutputFeatures  The number of output features
     * \return    Status object
     */
    virtual services::Status setNumberOfOutputFeatures(size_t numberOfOutputFeatures) = 0;

    /**
     * Sets type of the output feature
     * \param[in]  outputFeatureIndex  The output feature index
     * \param[in]  featureType         The type of the feature
     * \return     Status object
     */
    virtual services::Status setOutputFeatureType(size_t outputFeatureIndex,
                                                  features::FeatureType featureType) = 0;

    /**
     * Sets the number of categories for categorical feature
     * \param[in]  outputFeatureIndex  The index of output feature
     * \param[in]  numberOfCategories  The number of categories
     * \return     Status object
     */
    virtual services::Status setNumberOfCategories(size_t outputFeatureIndex,
                                                   size_t numberOfCategories) = 0;

    /**
     * Sets the dictionary for the categorical feature
     * \param[in]  outputFeatureIndex  The index of output feature
     * \param[in]  dictionary          The categorical feature dictionary
     * \return     Status object
     */
    virtual services::Status setCategoricalDictionary(size_t outputFeatureIndex,
                                                      const CategoricalFeatureDictionaryPtr &dictionary) = 0;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CONTEXTIFACE"></a>
 * \brief Abstract class that defines interface of modifier context
 */
class ContextIface
{
public:
    virtual ~ContextIface() { }

    /**
     * Gets the output buffer buffer. Buffer size must be equal to the number of
     * output features passed to the ConfigIface::setNumberOfOutputFeatures method.
     * By default buffer size is equal to the number of tokens
     * \return  The output buffer.
     */
    virtual services::BufferView<DAAL_DATA_TYPE> getOutputBuffer() const = 0;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__FEATUREMODIFIERIFACE"></a>
 * \brief  General feature modifier interface
 * \tparam Config Type of object to be used as configuration on initialization or finalization stages
 * \tparam Context Type of object that represents modifier context
 */
template<typename Config, typename Context>
class FeatureModifierIface
{
public:
    virtual ~FeatureModifierIface() { }

    /**
     * Executes initialization of the modifier. This method is
     * called once before FeatureModifierIface::apply
     * \param config  The configuration of the modifier
     */
    virtual void initialize(Config &config) = 0;

    /**
     * Applies the modifier. This method is supposed to extract the data from data
     * source provided via context and writes result to the buffer also provided via
     * context. The method may be called multiple times depending on data source
     * \param context  The context of the modifier
     */
    virtual void apply(Context &context) = 0;

    /**
     * Executes finalization of the modifier. This method is
     * called once after FeatureModifierIface::apply
     * \param config  The configuration of the modifier
     */
    virtual void finalize(Config &config) = 0;
};

/** @} */
} // namespace interface1

using interface1::ConfigIface;
using interface1::ContextIface;
using interface1::FeatureModifierIface;

} // namespace modifiers
} // namespace data_management
} // namespace daal

#endif
