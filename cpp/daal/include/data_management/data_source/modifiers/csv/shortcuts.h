/* file: shortcuts.h */
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
namespace interface1
{
/**
 * @ingroup data_source_modifiers_csv
 * @{
 */

/**
 * Simplifies creation of custom feature modifier
 * \tparam  Modifier  Type of modifier to be constructed
 * \return  Shared pointer to the modifier
 */
template <typename Modifier>
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
    return services::internal::wrapSharedAndTryThrow<FeatureModifier>(new internal::ContinuousFeatureModifier());
}

/**
 * Crates categorical feature modifier which interprets input tokens as
 * categorical features and replaces them with corresponding number
 * \return Shared pointer to the categorical feature modifier
 */
inline FeatureModifierIfacePtr categorical()
{
    return services::internal::wrapSharedAndTryThrow<FeatureModifier>(new internal::CategoricalFeatureModifier());
}

/**
 * Creates automatic feature modifier which automatically decides the best way to parse the feature
 * \return Shared pointer to the automatic feature modifier
 */
inline FeatureModifierIfacePtr automatic()
{
    return services::internal::wrapSharedAndTryThrow<FeatureModifier>(new internal::AutomaticFeatureModifier());
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
