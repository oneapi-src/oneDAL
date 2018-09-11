/* file: service_type_traits.h */
/*******************************************************************************
* Copyright 2015-2018 Intel Corporation.
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

#ifndef __SERVICE_TYPE_TRAITS_H__
#define __SERVICE_TYPE_TRAITS_H__

namespace daal
{
namespace services
{
namespace internal
{

template<CpuType cpu, typename T>
struct RemoveReference { typedef T type; };

template<CpuType cpu, typename T>
struct RemoveReference<cpu, T&> { typedef T type; };

template<CpuType cpu, typename T>
struct RemoveReference<cpu, T&&> { typedef T type; };


template<CpuType cpu, bool templateValue>
struct BoolConstant
{
#if (_MSC_VER <= 1800)
    static const bool value = templateValue;
#else
    static constexpr bool value = templateValue;
#endif
};

template<CpuType cpu>
using TrueConstant = BoolConstant<cpu, true>;

template<CpuType cpu>
using FalseConstant = BoolConstant<cpu, false>;

// Mark all types as non-primitive
template<typename T, CpuType cpu>
struct IsPrimitiveType : FalseConstant<cpu> { };

// Mark all pointer types as primitive
template<typename T, CpuType cpu>
struct IsPrimitiveType<T *, cpu> : TrueConstant<cpu> { };

#define __DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(type) \
    template<CpuType cpu> struct IsPrimitiveType<type, cpu> : TrueConstant<cpu> { };

// Mark built-in types as primitive
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE( bool               );
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE( char               );
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE( signed char        );
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE( unsigned char      );
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE( short              );
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE( unsigned short     );
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE( int                );
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE( unsigned int       );
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE( long               );
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE( unsigned long      );
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE( long long          );
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE( unsigned long long );
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE( float              );
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE( double             );
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE( long double        );

} // namespace internal
} // namespace services
} // namespace daal

#endif
