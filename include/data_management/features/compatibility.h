/* file: compatibility.h */
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

#ifndef __DATA_MANAGEMENT_FEATURES_COMPATIBILITY_H__
#define __DATA_MANAGEMENT_FEATURES_COMPATIBILITY_H__

#include "data_management/features/defines.h"
#include "data_management/features/internal/helpers.h"
#include "data_management/data/internal/conversion.h"

namespace daal
{
namespace data_management
{
/**
 * \brief Contains service functionality that simplifies feature handling
 */
namespace data_feature_utils
{

const int NumOfIndexNumTypes = (int)(data_management::features::DAAL_OTHER_T);

/* Usings for IndexNumType */
using data_management::features::IndexNumType;
using data_management::features::DAAL_FLOAT32;
using data_management::features::DAAL_FLOAT64;
using data_management::features::DAAL_INT32_S;
using data_management::features::DAAL_INT32_U;
using data_management::features::DAAL_INT64_S;
using data_management::features::DAAL_INT64_U;
using data_management::features::DAAL_INT8_S;
using data_management::features::DAAL_INT8_U;
using data_management::features::DAAL_INT16_S;
using data_management::features::DAAL_INT16_U;
using data_management::features::DAAL_OTHER_T;

/* Usings for PMMLNumType */
using data_management::features::PMMLNumType;
using data_management::features::DAAL_GEN_FLOAT;
using data_management::features::DAAL_GEN_DOUBLE;
using data_management::features::DAAL_GEN_INTEGER;
using data_management::features::DAAL_GEN_BOOLEAN;
using data_management::features::DAAL_GEN_STRING;
using data_management::features::DAAL_GEN_UNKNOWN;

/* Usings for FeatureType */
using data_management::features::FeatureType;
using data_management::features::DAAL_CATEGORICAL;
using data_management::features::DAAL_ORDINAL;
using data_management::features::DAAL_CONTINUOUS;

/* Usings for InternalNumType */
typedef data_management::internal::ConversionDataType InternalNumType;
using data_management::internal::DAAL_SINGLE;
using data_management::internal::DAAL_DOUBLE;
using data_management::internal::DAAL_INT32;
using data_management::internal::DAAL_OTHER;

/* Usings for helper functions */
using data_management::features::internal::getIndexNumType;
using data_management::features::internal::getPMMLNumType;
using data_management::internal::vectorConvertFuncType;
using data_management::internal::vectorStrideConvertFuncType;
DAAL_EXPORT vectorConvertFuncType getVectorUpCast(int, int);
DAAL_EXPORT vectorConvertFuncType getVectorDownCast(int, int);
DAAL_EXPORT vectorStrideConvertFuncType getVectorStrideUpCast(int, int);
DAAL_EXPORT vectorStrideConvertFuncType getVectorStrideDownCast(int, int);


template<typename T>
inline InternalNumType getInternalNumType()
{
    return data_management::internal::getConversionDataType<T>();
}

} // namespace data_feature_utils
} // namespace data_management
} // namespace daal

namespace DataFeatureUtils = daal::data_management::data_feature_utils;

#endif
