/* file: conversion.h */
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

#ifndef __DATA_MANAGEMENT_DATA_INTERNAL_CONVERSION_H__
#define __DATA_MANAGEMENT_DATA_INTERNAL_CONVERSION_H__

#include "data_management/features/defines.h"

namespace daal
{
namespace data_management
{
namespace internal
{

/* Renamed from InternalNumType */
enum ConversionDataType
{
    DAAL_SINGLE = 0,
    DAAL_DOUBLE = 1,
    DAAL_INT32  = 2,
    DAAL_OTHER  = 0xfffffff
};

/**
 * \return Internal numeric type
 */
template<typename T>
inline ConversionDataType getConversionDataType()          { return DAAL_OTHER;  }
template<>
inline ConversionDataType getConversionDataType<int>()     { return DAAL_INT32;  }
template<>
inline ConversionDataType getConversionDataType<double>()  { return DAAL_DOUBLE; }
template<>
inline ConversionDataType getConversionDataType<float>()   { return DAAL_SINGLE; }


typedef void(*vectorConvertFuncType)(size_t n, const void *src,
                                               void *dst);

typedef void(*vectorStrideConvertFuncType)(size_t n, const void *src, size_t srcByteStride,
                                                     void *dst, size_t dstByteStride);

DAAL_EXPORT vectorConvertFuncType getVectorUpCast(int, int);
DAAL_EXPORT vectorConvertFuncType getVectorDownCast(int, int);

DAAL_EXPORT vectorStrideConvertFuncType getVectorStrideUpCast(int, int);
DAAL_EXPORT vectorStrideConvertFuncType getVectorStrideDownCast(int, int);

} // namespace internal
} // namespace data_management
} // namespace daal

#endif
