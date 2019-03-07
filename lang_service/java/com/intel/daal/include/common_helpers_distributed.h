/* file: common_helpers_distributed.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

template<ComputeStep _Step, typename _Method, template<ComputeStep, typename, _Method> class _AlgClass, int head, int ... values >
struct jniDistributed
{
    template<typename... Args>
    static jlong newObj( jint prec, int method, Args&&... args )
    {
        if( method==head ) return jniDistributed<_Step, _Method, _AlgClass, head     >::newObj( prec, method, args... );
        else               return jniDistributed<_Step, _Method, _AlgClass, values...>::newObj( prec, method, args... );
    }

    static jlong getParameter( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniDistributed<_Step, _Method, _AlgClass, head     >::getParameter( prec, method, algAddr );
        else               return jniDistributed<_Step, _Method, _AlgClass, values...>::getParameter( prec, method, algAddr );
    }

    static jlong getInput( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniDistributed<_Step, _Method, _AlgClass, head     >::getInput( prec, method, algAddr );
        else               return jniDistributed<_Step, _Method, _AlgClass, values...>::getInput( prec, method, algAddr );
    }

    template<typename _Input>
    static void setInput( jint prec, int method, jlong algAddr, jlong inputAddr)
    {
        if( method==head ) jniDistributed<_Step, _Method, _AlgClass, head     >::template setInput<_Input>( prec, method, algAddr, inputAddr );
        else               jniDistributed<_Step, _Method, _AlgClass, values...>::template setInput<_Input>( prec, method, algAddr, inputAddr );
    }

    static jlong getResult( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniDistributed<_Step, _Method, _AlgClass, head     >::getResult( prec, method, algAddr );
        else               return jniDistributed<_Step, _Method, _AlgClass, values...>::getResult( prec, method, algAddr );
    }

    template<typename _Result>
    static void setResult( jint prec, int method, jlong algAddr, jlong resAddr )
    {
        if( method==head ) jniDistributed<_Step, _Method, _AlgClass, head     >::
            template setResult<_Result>( prec, method, algAddr, resAddr );
        else               jniDistributed<_Step, _Method, _AlgClass, values...>::
            template setResult<_Result>( prec, method, algAddr, resAddr );
    }

    static jlong getPartialResult( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniDistributed<_Step, _Method, _AlgClass, head     >::getPartialResult( prec, method, algAddr );
        else               return jniDistributed<_Step, _Method, _AlgClass, values...>::getPartialResult( prec, method, algAddr );
    }

    template<typename _PartialResult, typename... Args>
    static void setPartialResult( jint prec, int method, jlong algAddr, jlong presAddr, Args&&... args )
    {
        if( method==head ) jniDistributed<_Step, _Method, _AlgClass, head     >::
            template setPartialResult<_PartialResult>( prec, method, algAddr, presAddr, args... );
        else               jniDistributed<_Step, _Method, _AlgClass, values...>::
            template setPartialResult<_PartialResult>( prec, method, algAddr, presAddr, args... );
    }

    template<template <_Method> class _PartialResult, typename... Args>
    static void setPartialResultImpl( jint prec, int method, jlong algAddr, jlong presAddr, Args&&... args )
    {
        if( method==head ) jniDistributed<_Step, _Method, _AlgClass, head     >::
            template setPartialResultImpl<_PartialResult>( prec, method, algAddr, presAddr, args... );
        else               jniDistributed<_Step, _Method, _AlgClass, values...>::
            template setPartialResultImpl<_PartialResult>( prec, method, algAddr, presAddr, args... );
    }

    static jlong getClone( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniDistributed<_Step, _Method, _AlgClass, head     >::getClone( prec, method, algAddr );
        else               return jniDistributed<_Step, _Method, _AlgClass, values...>::getClone( prec, method, algAddr );
    }
};

template<ComputeStep _Step, typename _Method, template<ComputeStep, typename, _Method> class _AlgClass, int head>
struct jniDistributed<_Step, _Method, _AlgClass, head>
{
    template<typename... Args>
    static jlong newObj( jint prec, int method, Args&&... args )
    {
        if( method==head )
        {
            if(prec == 0)
            {
                return (jlong)(new SharedPtr<AlgorithmIface>(new _AlgClass<_Step, double, (_Method)head>( args... )));
            }
            else
            {
                return (jlong)(new SharedPtr<AlgorithmIface>(new _AlgClass<_Step, float, (_Method)head>( args... )));
            }
        }
        return 0;
    }

    static jlong getParameter( jint prec, int method, jlong algAddr )
    {
        if( method==head )
        {
            if(prec == 0)
            {
                return (jlong) & (staticPointerCast<_AlgClass<_Step,double,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr))->
                    parameter;
            }
            else
            {
                return (jlong) & (staticPointerCast<_AlgClass<_Step,float,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr))->
                    parameter;
            }
        }
        return 0;
    }

    static jlong getInput( jint prec, int method, jlong algAddr )
    {
        if( method==head )
        {
            if(prec == 0)
            {
                return (jlong) & staticPointerCast<_AlgClass<_Step,double,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->
                    input;
            }
            else
            {
                return (jlong) & staticPointerCast<_AlgClass<_Step,float,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->
                    input;
            }
        }
        return 0;
    }

    template<typename _Input>
    static void setInput( jint prec, int method, jlong algAddr, jlong inputAddr )
    {
        if( method==head )
        {
            if(prec == 0)
            {
                staticPointerCast<_AlgClass<_Step,double,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->input =
                    *((_Input*)inputAddr);
            }
            else
            {
                staticPointerCast<_AlgClass<_Step,float,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->input =
                    *((_Input*)inputAddr);
            }
        }
    }

    static jlong getResult( jint prec, int method, jlong algAddr )
    {
        SerializationIfacePtr *ptr = new SerializationIfacePtr();
        if( method==head )
        {
            if(prec == 0)
            {
                *ptr = staticPointerCast<_AlgClass<_Step,double,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->getResult();
            }
            else
            {
                *ptr = staticPointerCast<_AlgClass<_Step,float,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->getResult();
            }
        }
        return (jlong)ptr;
    }

    template<typename _Result>
    static void setResult( jint prec, int method, jlong algAddr, jlong resAddr )
    {
        SerializationIfacePtr *serializableShPtr = (SerializationIfacePtr *)resAddr;
        SharedPtr<_Result> resShPtr = staticPointerCast<_Result, SerializationIface>(*serializableShPtr);

        if( method==head )
        {
            if(prec == 0)
            {
                staticPointerCast<_AlgClass<_Step,double,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->setResult(resShPtr);
            }
            else
            {
                staticPointerCast<_AlgClass<_Step,float,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->setResult(resShPtr);
            }
        }
    }

    static jlong getPartialResult( jint prec, int method, jlong algAddr )
    {
        SerializationIfacePtr *ptr = new SerializationIfacePtr();
        if( method==head )
        {
            if(prec == 0)
            {
                *ptr = staticPointerCast<_AlgClass<_Step,double,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->
                    getPartialResult();
            }
            else
            {
                *ptr = staticPointerCast<_AlgClass<_Step,float,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->
                    getPartialResult();
            }
        }
        return (jlong)ptr;
    }

    template<typename _PartialResult, typename... Args>
    static void setPartialResult( jint prec, int method, jlong algAddr, jlong presAddr, Args&&... args )
    {
        SerializationIfacePtr *serializableShPtr = (SerializationIfacePtr *)presAddr;
        SharedPtr<_PartialResult> presShPtr = staticPointerCast<_PartialResult, SerializationIface>(*serializableShPtr);

        if( method==head )
        {
            if(prec == 0)
            {
                staticPointerCast<_AlgClass<_Step,double,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->
                    setPartialResult(presShPtr, args...);
            }
            else
            {
                staticPointerCast<_AlgClass<_Step,float,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->
                    setPartialResult(presShPtr, args...);
            }
        }
    }

    template<template <_Method> class _PartialResult, typename... Args>
    static void setPartialResultImpl( jint prec, int method, jlong algAddr, jlong presAddr, Args&&... args )
    {
        SerializationIfacePtr *serializableShPtr = (SerializationIfacePtr *)presAddr;

        if( method==head )
        {
            if(prec == 0)
            {
                SharedPtr<_PartialResult<(_Method)head> > presShPtr =
                    staticPointerCast<_PartialResult<(_Method)head>, SerializationIface>(*serializableShPtr);
                staticPointerCast<_AlgClass<_Step,double,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->
                    setPartialResult(presShPtr, args...);
            }
            else
            {
                SharedPtr<_PartialResult<(_Method)head> > presShPtr =
                    staticPointerCast<_PartialResult<(_Method)head>, SerializationIface>(*serializableShPtr);
                staticPointerCast<_AlgClass<_Step,float,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->
                    setPartialResult(presShPtr, args...);
            }
        }
    }

    static jlong getClone( jint prec, int method, jlong algAddr )
    {
        services::SharedPtr<AlgorithmIface> *ptr = new services::SharedPtr<AlgorithmIface>();
        if( method==head )
        {
            if(prec == 0)
            {
                *ptr = staticPointerCast<_AlgClass<_Step,double,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->
                    clone();
            }
            else
            {
                *ptr = staticPointerCast<_AlgClass<_Step,float,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->
                    clone();
            }
        }
        return (jlong)ptr;
    }
};

}
