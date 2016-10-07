/* file: data_utils.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#ifndef __DATA_UTILS_H__
#define __DATA_UTILS_H__

#include <string>
#include <climits>
#include <cfloat>
#include "services/daal_defines.h"

namespace daal
{
namespace data_management
{
namespace data_feature_utils
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
const int NumOfIndexNumTypes = (int)DAAL_OTHER_T;

enum InternalNumType  { DAAL_SINGLE = 0, DAAL_DOUBLE = 1, DAAL_INT32 = 2, DAAL_OTHER = 0xfffffff };
enum PMMLNumType      { DAAL_GEN_FLOAT = 0, DAAL_GEN_DOUBLE = 1, DAAL_GEN_INTEGER = 2, DAAL_GEN_BOOLEAN = 3,
                        DAAL_GEN_STRING = 4, DAAL_GEN_UNKNOWN = 0xfffffff
                      };
enum FeatureType      { DAAL_CATEGORICAL = 0, DAAL_ORDINAL = 1, DAAL_CONTINUOUS = 2 };

// Convert from a given C++ type to InternalNumType
template<typename T> inline IndexNumType getIndexNumType() { return DAAL_OTHER_T; }
template<> inline IndexNumType getIndexNumType<float>()            { return DAAL_FLOAT32; }
template<> inline IndexNumType getIndexNumType<double>()           { return DAAL_FLOAT64; }
template<> inline IndexNumType getIndexNumType<int>()              { return DAAL_INT32_S; }
template<> inline IndexNumType getIndexNumType<unsigned int>()     { return DAAL_INT32_U; }
template<> inline IndexNumType getIndexNumType<DAAL_INT64>()       { return DAAL_INT64_S; }
template<> inline IndexNumType getIndexNumType<DAAL_UINT64>()      { return DAAL_INT64_U; }
template<> inline IndexNumType getIndexNumType<char>()             { return DAAL_INT8_S;  }
template<> inline IndexNumType getIndexNumType<unsigned char>()    { return DAAL_INT8_U;  }
template<> inline IndexNumType getIndexNumType<short>()            { return DAAL_INT16_S; }
template<> inline IndexNumType getIndexNumType<unsigned short>()   { return DAAL_INT16_U; }

template<> inline IndexNumType getIndexNumType<long>()
{ return (IndexNumType)(DAAL_INT32_S + (sizeof(long) / 4 - 1) * 2); }

#if (defined(__APPLE__) || defined(__MACH__)) && !defined(__x86_64__)
template<> inline IndexNumType getIndexNumType<unsigned long>()
{ return (IndexNumType)(DAAL_INT32_U + (sizeof(unsigned long) / 4 - 1) * 2); }
#endif

#if !(defined(_WIN32) || defined(_WIN64)) && defined(__x86_64__)
template<> inline IndexNumType getIndexNumType<size_t>()
{ return (IndexNumType)(DAAL_INT32_U + (sizeof(size_t) / 4 - 1) * 2); }
#endif

template<typename T>
inline InternalNumType getInternalNumType()          { return DAAL_OTHER;  }
template<>
inline InternalNumType getInternalNumType<int>()     { return DAAL_INT32;  }
template<>
inline InternalNumType getInternalNumType<double>()  { return DAAL_DOUBLE; }
template<>
inline InternalNumType getInternalNumType<float>()   { return DAAL_SINGLE; }

template<typename T>
inline PMMLNumType getPMMLNumType()                { return DAAL_GEN_UNKNOWN; }
template<>
inline PMMLNumType getPMMLNumType<int>()           { return DAAL_GEN_INTEGER; }
template<>
inline PMMLNumType getPMMLNumType<double>()        { return DAAL_GEN_DOUBLE;  }
template<>
inline PMMLNumType getPMMLNumType<float>()         { return DAAL_GEN_FLOAT;   }
template<>
inline PMMLNumType getPMMLNumType<bool>()          { return DAAL_GEN_BOOLEAN; }
template<>
inline PMMLNumType getPMMLNumType<char *>()         { return DAAL_GEN_STRING;  }
template<>
inline PMMLNumType getPMMLNumType<std::string>()   { return DAAL_GEN_STRING;  }

template<typename T>
inline T      getMaxVal()          { return 0;  }
template<>
inline int    getMaxVal<int>()     { return INT_MAX; }
template<>
inline double getMaxVal<double>()  { return DBL_MAX; }
template<>
inline float  getMaxVal<float>()   { return FLT_MAX; }

template<typename T>
inline T      getMinVal()          { return 0;  }
template<>
inline int    getMinVal<int>()     { return INT_MIN; }
template<>
inline double getMinVal<double>()  { return DBL_MIN; }
template<>
inline float  getMinVal<float>()   { return FLT_MIN; }

template<typename T>
inline T      getEpsilonVal()          { return 0;  }
template<>
inline double getEpsilonVal<double>()  { return DBL_EPSILON; }
template<>
inline float  getEpsilonVal<float>()   { return FLT_EPSILON; }

typedef void(*vectorConvertFuncType)(size_t n, void *src, void *dst);
typedef void(*vectorStrideConvertFuncType)(size_t n, void *src, size_t srcByteStride, void *dst, size_t dstByteStride);

template<typename T1, typename T2>
static void vectorConvertFunc(size_t n, void *src, void *dst)
{
    for(size_t i = 0; i < n; i++)
    {
        ((T2 *)dst)[i] = static_cast<T2>(((T1 *)src)[i]);
    }
}

template<typename T1, typename T2>
static void vectorStrideConvertFunc(size_t n, void *src, size_t srcByteStride, void *dst, size_t dstByteStride)
{
    for(size_t i = 0; i < n ; i++)
    {
        *(T2 *)(((char *)dst) + i * dstByteStride) = static_cast<T2>(*(T1 *)(((char *)src) + i * srcByteStride));
    }
}

#undef  DAAL_TABLE_UP_ENTRY
#define DAAL_TABLE_UP_ENTRY(F,T) {F<T, float>, F<T, double>, F<T, int> }

#undef  DAAL_TABLE_DOWN_ENTRY
#define DAAL_TABLE_DOWN_ENTRY(F,T) {F<float, T>, F<double, T>, F<int, T> }

#undef  DAAL_CONVERT_UP_TABLE
#define DAAL_CONVERT_UP_TABLE(F) {              \
        DAAL_TABLE_UP_ENTRY(F,float),               \
        DAAL_TABLE_UP_ENTRY(F,double),              \
        DAAL_TABLE_UP_ENTRY(F,int),                 \
        DAAL_TABLE_UP_ENTRY(F,unsigned int),        \
        DAAL_TABLE_UP_ENTRY(F,DAAL_INT64),          \
        DAAL_TABLE_UP_ENTRY(F,DAAL_UINT64),         \
        DAAL_TABLE_UP_ENTRY(F,char),                \
        DAAL_TABLE_UP_ENTRY(F,unsigned char),       \
        DAAL_TABLE_UP_ENTRY(F,short),               \
        DAAL_TABLE_UP_ENTRY(F,unsigned short),      \
    }

#undef  DAAL_CONVERT_DOWN_TABLE
#define DAAL_CONVERT_DOWN_TABLE(F) {           \
        DAAL_TABLE_DOWN_ENTRY(F,float),            \
        DAAL_TABLE_DOWN_ENTRY(F,double),           \
        DAAL_TABLE_DOWN_ENTRY(F,int),              \
        DAAL_TABLE_DOWN_ENTRY(F,unsigned int),     \
        DAAL_TABLE_DOWN_ENTRY(F,DAAL_INT64),       \
        DAAL_TABLE_DOWN_ENTRY(F,DAAL_UINT64),      \
        DAAL_TABLE_DOWN_ENTRY(F,char),             \
        DAAL_TABLE_DOWN_ENTRY(F,unsigned char),    \
        DAAL_TABLE_DOWN_ENTRY(F,short),            \
        DAAL_TABLE_DOWN_ENTRY(F,unsigned short),   \
    }

static data_feature_utils::vectorConvertFuncType vectorUpCast[NumOfIndexNumTypes][3] = DAAL_CONVERT_UP_TABLE(vectorConvertFunc);
static data_feature_utils::vectorConvertFuncType vectorDownCast[NumOfIndexNumTypes][3] = DAAL_CONVERT_DOWN_TABLE(vectorConvertFunc);

static data_feature_utils::vectorStrideConvertFuncType vectorStrideUpCast[NumOfIndexNumTypes][3] = DAAL_CONVERT_UP_TABLE(vectorStrideConvertFunc);
static data_feature_utils::vectorStrideConvertFuncType vectorStrideDownCast[NumOfIndexNumTypes][3] = DAAL_CONVERT_DOWN_TABLE(vectorStrideConvertFunc);
/** @} */

} // namespace data_feature_utils
#define DataFeatureUtils data_feature_utils
}
} // namespace daal
#endif
