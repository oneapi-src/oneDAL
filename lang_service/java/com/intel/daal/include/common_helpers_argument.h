/* file: common_helpers_argument.h */
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
        SerializationIfacePtr * dShPtr = new SerializationIfacePtr();
        *dShPtr = staticPointerCast<SerializationIface, _DataType>( argument->get( (_IdType)id , args... ) );
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
