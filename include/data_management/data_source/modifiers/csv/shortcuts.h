/* file: shortcuts.h */
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

#ifndef __DATA_SOURCE_MODIFIERS_CSV_SHORTCUTS_H__
#define __DATA_SOURCE_MODIFIERS_CSV_SHORTCUTS_H__

#include "services/internal/error_handling_helpers.h"
#include "data_management/data_source/modifiers/csv/internal/default_modifiers.h"

namespace daal
{
namespace data_management
{
namespace modifiers
{
namespace csv
{
namespace interface1 {

/**
 * @ingroup data_source_modifiers_csv
 * @{
 */

/**
 * Simplifies creation of custom feature modifier
 * \tparam  Modifier  Type of modifier to be constructed
 * \return  Shared pointer to the modifier
 */
template<typename Modifier>
inline FeatureModifierIfacePtr custom()
{
    return services::internal::wrapSharedAndTryThrow<Modifier>(new Modifier());
}

/**
 * Creates continuous feature modifier which parses input tokens as real numbers
 * \return Shared pointer to the continuous feature modifier
 */
inline FeatureModifierIfacePtr continuous()
{
    return services::internal::wrapSharedAndTryThrow<FeatureModifier>(
        new internal::ContinuousFeatureModifier());
}

/**
 * Crates categorical feature modifier which interprets input tokens as
 * categorical features and replaces them with corresponding number
 * \return Shared pointer to the categorical feature modifier
 */
inline FeatureModifierIfacePtr categorical()
{
    return services::internal::wrapSharedAndTryThrow<FeatureModifier>(
        new internal::CategoricalFeatureModifier());
}

/**
 * Creates automatic feature modifier which automatically decides the best way to parse the feature
 * \return Shared pointer to the automatic feature modifier
 */
inline FeatureModifierIfacePtr automatic()
{
    return services::internal::wrapSharedAndTryThrow<FeatureModifier>(
        new internal::AutomaticFeatureModifier());
}

/** @} */
} // namespace interface1

using interface1::custom;
using interface1::continuous;
using interface1::categorical;
using interface1::automatic;

} // namespace csv
} // namespace modifiers
} // namespace data_management
} // namespace daal

#endif
