/* file: conversion.h */
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

#ifndef __DATA_MANAGEMENT_DATA_INTERNAL_CONVERSION_H__
#define __DATA_MANAGEMENT_DATA_INTERNAL_CONVERSION_H__

#include "data_management/features/defines.h"
#include "data_management/features/internal/helpers.h"

namespace daal
{
namespace data_management
{
namespace internal
{
/**
 * @defgroup data_management_internal DataManagementInternal
 * \brief Internal classes of data management
 * @{
 */

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
template <typename T>
inline ConversionDataType getConversionDataType()
{
    return DAAL_OTHER;
}
template <>
inline ConversionDataType getConversionDataType<int>()
{
    return DAAL_INT32;
}
template <>
inline ConversionDataType getConversionDataType<double>()
{
    return DAAL_DOUBLE;
}
template <>
inline ConversionDataType getConversionDataType<float>()
{
    return DAAL_SINGLE;
}

typedef void (*vectorConvertFuncType)(size_t n, const void * src, void * dst);
typedef void (*vectorStrideConvertFuncType)(size_t n, const void * src, size_t srcByteStride, void * dst, size_t dstByteStride);

typedef bool (*vectorCopy2vFuncType)(const size_t nrows, const size_t ncols, void * dst, void const * ptrMin, DAAL_INT64 const * arrOffsets);

template <typename T>
DAAL_EXPORT vectorCopy2vFuncType getVector();

template <>
DAAL_EXPORT vectorCopy2vFuncType getVector<int>();
template <>
DAAL_EXPORT vectorCopy2vFuncType getVector<float>();
template <>
DAAL_EXPORT vectorCopy2vFuncType getVector<double>();

DAAL_EXPORT vectorConvertFuncType getVectorUpCast(int, int);
DAAL_EXPORT vectorConvertFuncType getVectorDownCast(int, int);

DAAL_EXPORT vectorStrideConvertFuncType getVectorStrideUpCast(int, int);
DAAL_EXPORT vectorStrideConvertFuncType getVectorStrideDownCast(int, int);

/**
 *  <a name="DAAL-CLASS-DATAMANAGEMENT-INTERNAL__VECTORUPCAST"></a>
 *  \brief Class to cast vector up from T type to U
 */
template <typename T, typename U>
class VectorUpCast
{
public:
    typedef T SourceType;
    typedef U DestType;

    VectorUpCast() : _func(getVectorUpCast(features::internal::getIndexNumType<T>(), getConversionDataType<U>())) {}

    void operator()(size_t size, const T * src, U * dst) const { _func(size, src, dst); }

private:
    vectorConvertFuncType _func;
};

/**
 *  <a name="DAAL-CLASS-DATAMANAGEMENT-INTERNAL__VECTORDOWNCAST"></a>
 *  \brief Class to cast vector down from T type to U
 */
template <typename T, typename U>
class VectorDownCast
{
public:
    typedef T SourceType;
    typedef U DestType;

    VectorDownCast() : _func(getVectorDownCast(features::internal::getIndexNumType<U>(), getConversionDataType<T>())) {}

    void operator()(size_t size, const T * src, U * dst) const { _func(size, src, dst); }

private:
    vectorConvertFuncType _func;
};

#define DAAL_REGISTER_WITH_HOMOGEN_NT_TYPES(FUNC) \
    FUNC(float)                                   \
    FUNC(double)                                  \
    FUNC(int)                                     \
    FUNC(unsigned int)                            \
    FUNC(DAAL_INT64)                              \
    FUNC(DAAL_UINT64)                             \
    FUNC(char)                                    \
    FUNC(unsigned char)                           \
    FUNC(short)                                   \
    FUNC(unsigned short)                          \
    FUNC(long)                                    \
    FUNC(unsigned long)

template <typename T>
DAAL_EXPORT void vectorAssignValueToArray(T * const ptr, const size_t n, const T value);

/** @} */

} // namespace internal
} // namespace data_management
} // namespace daal

#endif
