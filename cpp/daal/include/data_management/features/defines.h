/* file: defines.h */
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
//  Implementation of data dictionary utilities.
//--
*/

#ifndef __DATA_MANAGEMENT_FEATURES_DEFINES_H__
#define __DATA_MANAGEMENT_FEATURES_DEFINES_H__

#include <string>
#include <climits>
#include <cfloat>
#include <limits>

#include "services/daal_defines.h"

namespace daal
{
namespace data_management
{
/**
 * \brief Contains service functionality that simplifies feature handling
 */
namespace features
{
/**
 * @ingroup data_model
 * @{
 */

enum IndexNumType
{
    DAAL_FLOAT32 = 0,
    DAAL_FLOAT64 = 1,
    DAAL_INT32_S = 2,
    DAAL_INT32_U = 3,
    DAAL_INT64_S = 4,
    DAAL_INT64_U = 5,
    DAAL_INT8_S  = 6,
    DAAL_INT8_U  = 7,
    DAAL_INT16_S = 8,
    DAAL_INT16_U = 9,
    DAAL_OTHER_T = 10
};

/**
 * \return IndexNumType numeric type
 */
template <typename T>
inline IndexNumType getIndexNumType()
{
    return DAAL_OTHER_T;
}
template <>
inline IndexNumType getIndexNumType<int>()
{
    return DAAL_INT32_S;
}
template <>
inline IndexNumType getIndexNumType<double>()
{
    return DAAL_FLOAT64;
}
template <>
inline IndexNumType getIndexNumType<float>()
{
    return DAAL_FLOAT32;
}

enum PMMLNumType
{
    DAAL_GEN_FLOAT   = 0,
    DAAL_GEN_DOUBLE  = 1,
    DAAL_GEN_INTEGER = 2,
    DAAL_GEN_BOOLEAN = 3,
    DAAL_GEN_STRING  = 4,
    DAAL_GEN_UNKNOWN = 0xfffffff
};

enum FeatureType
{
    DAAL_CATEGORICAL = 0,
    DAAL_ORDINAL     = 1,
    DAAL_CONTINUOUS  = 2
};

/** @} */

} // namespace features
} // namespace data_management
} // namespace daal

#endif
