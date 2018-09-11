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

#ifndef __DATA_SOURCE_MODIFIERS_SQL_SHORTCUTS_H__
#define __DATA_SOURCE_MODIFIERS_SQL_SHORTCUTS_H__

#include "services/internal/error_handling_helpers.h"
#include "data_management/data_source/modifiers/sql/internal/default_modifiers.h"

namespace daal
{
namespace data_management
{
namespace modifiers
{
namespace sql
{
namespace interface1
{

/**
 * @ingroup data_source_modifiers_sql
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
 * Creates continuous feature modifier which parses input column values as real numbers
 * \return Shared pointer to the continuous feature modifier
 */
inline FeatureModifierIfacePtr continuous()
{
    return services::internal::wrapSharedAndTryThrow<FeatureModifier>(
        new internal::ContinuousFeatureModifier());
}

/** @} */
} // namespace interface1

using interface1::custom;
using interface1::continuous;

} // namespace sql
} // namespace modifiers
} // namespace data_management
} // namespace daal

#endif
