/* file: defines.h */
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
