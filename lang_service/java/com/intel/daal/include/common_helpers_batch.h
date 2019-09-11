/* file: common_helpers_batch.h */
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

using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;

template<typename _Method, template<typename, _Method> class _AlgClass, int head, int ... values >
struct jniBatch
{
    template<typename... Args>
    static jlong newObj( jint prec, int method, Args&&... args )
    {
        if( method==head ) return jniBatch<_Method, _AlgClass, head     >::newObj( prec, method, args... );
        else               return jniBatch<_Method, _AlgClass, values...>::newObj( prec, method, args... );
    }

    static jlong getParameter( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniBatch<_Method, _AlgClass, head     >::getParameter( prec, method, algAddr );
        else               return jniBatch<_Method, _AlgClass, values...>::getParameter( prec, method, algAddr );
    }

    static jlong getInput( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniBatch<_Method, _AlgClass, head     >::getInput( prec, method, algAddr );
        else               return jniBatch<_Method, _AlgClass, values...>::getInput( prec, method, algAddr );
    }

    template<typename _Input>
    static void setInput( jint prec, int method, jlong algAddr, jlong inputAddr )
    {
        if( method==head ) jniBatch<_Method, _AlgClass, head     >::template setInput<_Input>( prec, method, algAddr, inputAddr );
        else               jniBatch<_Method, _AlgClass, values...>::template setInput<_Input>( prec, method, algAddr, inputAddr );
    }

    static jlong getResult( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniBatch<_Method, _AlgClass, head     >::getResult( prec, method, algAddr );
        else               return jniBatch<_Method, _AlgClass, values...>::getResult( prec, method, algAddr );
    }

    template<typename _Result>
    static void setResult( jint prec, int method, jlong algAddr, jlong resAddr )
    {
        if( method==head ) jniBatch<_Method, _AlgClass, head     >::template setResult<_Result>( prec, method, algAddr, resAddr );
        else               jniBatch<_Method, _AlgClass, values...>::template setResult<_Result>( prec, method, algAddr, resAddr );
    }

    static jlong getClone( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniBatch<_Method, _AlgClass, head     >::getClone( prec, method, algAddr );
        else               return jniBatch<_Method, _AlgClass, values...>::getClone( prec, method, algAddr );
    }
};

template<typename _Method, template<typename, _Method> class _AlgClass, int head>
struct jniBatch<_Method, _AlgClass, head>
{
    template<typename... Args>
    static jlong newObj( jint prec, int method, Args&&... args )
    {
        if( method==head )
        {
            if(prec == 0)
            {
                return (jlong)(new SharedPtr<AlgorithmIface>(new _AlgClass<double, (_Method)head>( args... )));
            }
            else
            {
                return (jlong)(new SharedPtr<AlgorithmIface>(new _AlgClass<float, (_Method)head>( args... )));
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
                return (jlong) (staticPointerCast<_AlgClass<double,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr))->
                    getBaseParameter();
            }
            else
            {
                return (jlong) (staticPointerCast<_AlgClass<float,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr))->
                    getBaseParameter();
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
                return (jlong) & staticPointerCast<_AlgClass<double,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->
                    input;
            }
            else
            {
                return (jlong) & staticPointerCast<_AlgClass<float,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->
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
                staticPointerCast<_AlgClass<double,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->input =
                    *(_Input*)inputAddr;
            }
            else
            {
                staticPointerCast<_AlgClass<float,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->input =
                    *(_Input*)inputAddr;
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
                *ptr = staticPointerCast<_AlgClass<double,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->getResult();
            }
            else
            {
                *ptr = staticPointerCast<_AlgClass<float,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->getResult();
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
                staticPointerCast<_AlgClass<double,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->setResult(resShPtr);
            }
            else
            {
                staticPointerCast<_AlgClass<float,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->setResult(resShPtr);
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
                *ptr = staticPointerCast<_AlgClass<double,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->clone();
            }
            else
            {
                *ptr = staticPointerCast<_AlgClass<float,(_Method)head>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->clone();
            }
        }
        return (jlong)ptr;
    }
};

template<typename _PMethod, typename _TMethod, _TMethod tmethod, template<typename, _PMethod, _TMethod> class _AlgClass, int head, int ... values >
struct jniBatchClassifier
{
    template<typename... Args>
    static jlong newObj( jint prec, int method, Args&&... args )
    {
        if( method==head ) return jniBatchClassifier<_PMethod, _TMethod, tmethod, _AlgClass, head     >::newObj( prec, method, args... );
        else               return jniBatchClassifier<_PMethod, _TMethod, tmethod, _AlgClass, values...>::newObj( prec, method, args... );
    }

    static jlong getParameter( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniBatchClassifier<_PMethod, _TMethod, tmethod, _AlgClass, head     >::getParameter( prec, method, algAddr );
        else               return jniBatchClassifier<_PMethod, _TMethod, tmethod, _AlgClass, values...>::getParameter( prec, method, algAddr );
    }

    static jlong getInput( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniBatchClassifier<_PMethod, _TMethod, tmethod, _AlgClass, head     >::getInput( prec, method, algAddr );
        else               return jniBatchClassifier<_PMethod, _TMethod, tmethod, _AlgClass, values...>::getInput( prec, method, algAddr );
    }

    template<typename _Input>
    static jlong setInput( jint prec, int method, jlong algAddr, jlong inputAddr )
    {
        if( method==head ) return jniBatchClassifier<_PMethod, _TMethod, tmethod, _AlgClass, head     >::
                                    template setInput<_Input>( prec, method, algAddr, inputAddr );
        else               return jniBatchClassifier<_PMethod, _TMethod, tmethod, _AlgClass, values...>::
                                    template setInput<_Input>( prec, method, algAddr, inputAddr );
    }

    static jlong getResult( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniBatchClassifier<_PMethod, _TMethod, tmethod, _AlgClass, head     >::getResult( prec, method, algAddr );
        else               return jniBatchClassifier<_PMethod, _TMethod, tmethod, _AlgClass, values...>::getResult( prec, method, algAddr );
    }

    template<typename _Result>
    static void setResult( jint prec, int method, jlong algAddr, jlong resAddr )
    {
        if( method==head ) jniBatchClassifier<_PMethod, _TMethod, tmethod, _AlgClass, head     >::
                                    template setResult<_Result>( prec, method, algAddr, resAddr );
        else               jniBatchClassifier<_PMethod, _TMethod, tmethod, _AlgClass, values...>::
                                    template setResult<_Result>( prec, method, algAddr, resAddr );
    }

    static jlong getClone( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniBatchClassifier<_PMethod, _TMethod, tmethod, _AlgClass, head     >::getClone( prec, method, algAddr );
        else               return jniBatchClassifier<_PMethod, _TMethod, tmethod, _AlgClass, values...>::getClone( prec, method, algAddr );
    }
};

template<typename _PMethod, typename _TMethod, _TMethod tmethod, template<typename, _PMethod, _TMethod> class _AlgClass, int head>
struct jniBatchClassifier<_PMethod, _TMethod, tmethod, _AlgClass, head>
{
    template<typename... Args>
    static jlong newObj( jint prec, int method, Args&&... args )
    {
        if( method==head )
        {
            if(prec == 0)
            {
                return (jlong)(new SharedPtr<AlgorithmIface>(new _AlgClass<double, (_PMethod)head, tmethod>( args... )));
            }
            else
            {
                return (jlong)(new SharedPtr<AlgorithmIface>(new _AlgClass<float, (_PMethod)head, tmethod>( args... )));
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
                return (jlong) & (staticPointerCast<_AlgClass<double,(_PMethod)head, tmethod>, AlgorithmIface>
                        (*(SharedPtr<AlgorithmIface> *)algAddr))->parameter;
            }
            else
            {
                return (jlong) & (staticPointerCast<_AlgClass<float,(_PMethod)head, tmethod>, AlgorithmIface>
                    (*(SharedPtr<AlgorithmIface> *)algAddr))->parameter;
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
                return (jlong) & staticPointerCast<_AlgClass<double,(_PMethod)head, tmethod>, AlgorithmIface>
                    (*(SharedPtr<AlgorithmIface> *)algAddr)->input;
            }
            else
            {
                return (jlong) & staticPointerCast<_AlgClass<float,(_PMethod)head, tmethod>, AlgorithmIface>
                    (*(SharedPtr<AlgorithmIface> *)algAddr)->input;
            }
        }
        return 0;
    }

    template<typename _Input>
    static jlong setInput( jint prec, int method, jlong algAddr, jlong inputAddr )
    {
        if( method==head )
        {
            if(prec == 0)
            {
                staticPointerCast<_AlgClass<double,(_PMethod)head, tmethod>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->input =
                    *(_Input*)inputAddr;
            }
            else
            {
                staticPointerCast<_AlgClass<float,(_PMethod)head, tmethod>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->input =
                    *(_Input*)inputAddr;
            }
        }
        return 0;
    }

    static jlong getResult( jint prec, int method, jlong algAddr )
    {
        SerializationIfacePtr *ptr = new SerializationIfacePtr();
        if( method==head )
        {
            if(prec == 0)
            {
                *ptr = staticPointerCast<_AlgClass<double,(_PMethod)head, tmethod>, AlgorithmIface>
                        (*(SharedPtr<AlgorithmIface> *)algAddr)->getResult();
            }
            else
            {
                *ptr = staticPointerCast<_AlgClass<float,(_PMethod)head, tmethod>, AlgorithmIface>
                        (*(SharedPtr<AlgorithmIface> *)algAddr)->getResult();
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
                staticPointerCast<_AlgClass<double,(_PMethod)head, tmethod>, AlgorithmIface>
                        (*(SharedPtr<AlgorithmIface> *)algAddr)->setResult(resShPtr);
            }
            else
            {
                staticPointerCast<_AlgClass<float,(_PMethod)head, tmethod>, AlgorithmIface>
                        (*(SharedPtr<AlgorithmIface> *)algAddr)->setResult(resShPtr);
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
                *ptr = staticPointerCast<_AlgClass<double,(_PMethod)head, tmethod>, AlgorithmIface>
                        (*(SharedPtr<AlgorithmIface> *)algAddr)->clone();
            }
            else
            {
                *ptr = staticPointerCast<_AlgClass<float,(_PMethod)head, tmethod>, AlgorithmIface>
                        (*(SharedPtr<AlgorithmIface> *)algAddr)->clone();
            }
        }
        return (jlong)ptr;
    }
};

template<typename _Method, template<typename, _Method> class _AlgClass, int head, int ... values >
struct jniBatchLayer
{
    template<typename... Args>
    static jlong newObj( jint prec, int method, Args&&... args )
    {
        if( method==head ) return jniBatchLayer<_Method, _AlgClass, head     >::newObj( prec, method, args... );
        else               return jniBatchLayer<_Method, _AlgClass, values...>::newObj( prec, method, args... );
    }

    static jlong getParameter( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniBatchLayer<_Method, _AlgClass, head     >::getParameter( prec, method, algAddr );
        else               return jniBatchLayer<_Method, _AlgClass, values...>::getParameter( prec, method, algAddr );
    }

    static jlong getForwardLayer( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniBatchLayer<_Method, _AlgClass, head     >::getForwardLayer( prec, method, algAddr );
        else               return jniBatchLayer<_Method, _AlgClass, values...>::getForwardLayer( prec, method, algAddr );
    }

    static jlong getBackwardLayer( jint prec, int method, jlong algAddr )
    {
        if( method==head ) return jniBatchLayer<_Method, _AlgClass, head     >::getBackwardLayer( prec, method, algAddr );
        else               return jniBatchLayer<_Method, _AlgClass, values...>::getBackwardLayer( prec, method, algAddr );
    }
};

template<typename _Method, template<typename, _Method> class _AlgClass, int head>
struct jniBatchLayer<_Method, _AlgClass, head>
{
    template<typename... Args>
    static jlong newObj( jint prec, int method, Args&&... args )
    {
        if( method==head )
        {
            if(prec == 0)
            {
                return (jlong)(new neural_networks::layers::LayerIfacePtr(new _AlgClass<double, (_Method)head>( args... )));
            }
            else
            {
                return (jlong)(new neural_networks::layers::LayerIfacePtr(new _AlgClass<float, (_Method)head>( args... )));
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
                return (jlong) & ((staticPointerCast<_AlgClass<double,(_Method)head>, neural_networks::layers::LayerIface>
                        (*(neural_networks::layers::LayerIfacePtr *)algAddr))->parameter);
            }
            else
            {
                return (jlong) & ((staticPointerCast<_AlgClass<float,(_Method)head>, neural_networks::layers::LayerIface>
                        (*(neural_networks::layers::LayerIfacePtr *)algAddr))->parameter);
            }
        }
        return 0;
    }

    static jlong getForwardLayer( jint prec, int method, jlong algAddr )
    {
        if( method==head )
        {
            if(prec == 0)
            {
                return (jlong) (new neural_networks::layers::forward::LayerIfacePtr(
                    (staticPointerCast<_AlgClass<double,(_Method)head>, neural_networks::layers::LayerIface>
                        (*(neural_networks::layers::LayerIfacePtr *)algAddr))->forwardLayer));
            }
            else
            {
                return (jlong) (new neural_networks::layers::forward::LayerIfacePtr(
                    (staticPointerCast<_AlgClass<float,(_Method)head>, neural_networks::layers::LayerIface>
                        (*(neural_networks::layers::LayerIfacePtr *)algAddr))->forwardLayer));
            }
        }
        return 0;
    }

    static jlong getBackwardLayer( jint prec, int method, jlong algAddr )
    {
        if( method==head )
        {
            if(prec == 0)
            {
                return (jlong) (new neural_networks::layers::backward::LayerIfacePtr(
                    (staticPointerCast<_AlgClass<double,(_Method)head>, neural_networks::layers::LayerIface>
                        (*(neural_networks::layers::LayerIfacePtr *)algAddr))->backwardLayer));
            }
            else
            {
                return (jlong) (new neural_networks::layers::backward::LayerIfacePtr(
                    (staticPointerCast<_AlgClass<float,(_Method)head>, neural_networks::layers::LayerIface>
                        (*(neural_networks::layers::LayerIfacePtr *)algAddr))->backwardLayer));
            }
        }
        return 0;
    }
};
}
