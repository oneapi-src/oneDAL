/* file: common_helpers_argument.h */
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

#include "daal.h"

namespace daal
{

using namespace daal::services;
using namespace daal::algorithms;
using namespace daal::data_management;

template<typename _Input>
struct jniInput
{
    template<typename _IdType, typename _DataType, typename... Args>
    static jlong get( jlong inputAddr, jint id, Args&&... args)
    {
        _Input *input = (_Input*)inputAddr;
        SerializationIfacePtr * dShPtr = new SerializationIfacePtr();
        *dShPtr = staticPointerCast<SerializationIface, _DataType>( input->get( (_IdType)id, args... ) );
        return (jlong)dShPtr;
    }

    template<typename _IdType, typename _DataType, typename... Args>
    static void set( jlong inputAddr, jint id, jlong dataAddr, Args&&... args )
    {
        _Input *input = (_Input*)inputAddr;
        SerializationIfacePtr *dataShPtr = (SerializationIfacePtr *)dataAddr;
        input->set((_IdType)id, staticPointerCast<_DataType, SerializationIface>(*dataShPtr), args...);
    }

    template<typename _IdType, typename _DataType>
    static void add( jlong inputAddr, jint id, jlong dataAddr )
    {
        _Input *input = (_Input*)inputAddr;
        SerializationIfacePtr *dataShPtr = (SerializationIfacePtr *)dataAddr;
        input->add((_IdType)id, staticPointerCast<_DataType, SerializationIface>(*dataShPtr));
    }

    template<typename _IdType, typename _DataType>
    static void add( jlong inputAddr, jint id, jint key, jlong dataAddr )
    {
        _Input *input = (_Input*)inputAddr;
        SerializationIfacePtr *dataShPtr = (SerializationIfacePtr *)dataAddr;
        input->add((_IdType)id, (size_t)key, staticPointerCast<_DataType, SerializationIface>(*dataShPtr));
    }

    template<typename _WidType, typename _IdType, typename _DataType>
    static void setex( jlong inputAddr, jint wid, jint id, jlong dataAddr )
    {
        _Input *input = (_Input*)inputAddr;
        SerializationIfacePtr *dataShPtr = (SerializationIfacePtr *)dataAddr;
        input->set((_WidType)wid, (_IdType)id, staticPointerCast<_DataType, SerializationIface>(*dataShPtr));
    }

    template<typename _WidType, typename _IdType, typename _DataType>
    static jlong getex( jlong inputAddr, jint wid, jint id )
    {
        _Input *input = (_Input*)inputAddr;
        SerializationIfacePtr * dShPtr = new SerializationIfacePtr();
        *dShPtr = staticPointerCast<SerializationIface, _DataType>( input->get( (_WidType)wid, (_IdType)id ) );
        return (jlong)dShPtr;
    }
};

template<typename _Argument>
struct jniArgument
{
    static jlong newObj()
    {
        return (jlong)( new SerializationIfacePtr( new _Argument() ) );
    }

    template<typename _IdType, typename _DataType, typename... Args>
    static jlong get( jlong argumentAddr, jint id, Args&&... args )
    {
        SharedPtr<_Argument> argument = staticPointerCast<_Argument, SerializationIface>( *(SerializationIfacePtr *)argumentAddr );
        services::SharedPtr<_DataType> tmp = argument->get((_IdType)id, args...);
        if(!tmp)
            return (jlong)0;
        SerializationIfacePtr * dShPtr = new SerializationIfacePtr;
        *dShPtr = staticPointerCast<SerializationIface, _DataType>(tmp);
        return (jlong)dShPtr;
    }

    template<typename _IdType, typename _DataType, typename... Args>
    static void set( jlong argumentAddr, jint id, jlong dataAddr, Args&&... args )
    {
        SharedPtr<_Argument> argument = staticPointerCast<_Argument, SerializationIface>( *(SerializationIfacePtr *)argumentAddr );
        SerializationIfacePtr *dataShPtr = (SerializationIfacePtr *)dataAddr;
        argument->set((_IdType)id, staticPointerCast<_DataType, SerializationIface>(*dataShPtr), args...);
    }

    template<typename _IdType, typename _DataType>
    static void add( jlong argumentAddr, jint id, jlong dataAddr )
    {
        SharedPtr<_Argument> argument = staticPointerCast<_Argument, SerializationIface>( *(SerializationIfacePtr *)argumentAddr );
        SerializationIfacePtr *dataShPtr = (SerializationIfacePtr *)dataAddr;
        argument->add((_IdType)id, staticPointerCast<_DataType, SerializationIface>(*dataShPtr));
    }
};

}
