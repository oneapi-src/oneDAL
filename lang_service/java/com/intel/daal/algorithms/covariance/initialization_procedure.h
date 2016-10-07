/* file: initialization_procedure.h */
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
//  Implementation of the class that connects Covariance Java initialization procedure
//  to C++ algorithm
//--
*/
#ifndef __INITIALIZATION_PROCEDURE_H__
#define __INITIALIZATION_PROCEDURE_H__

#include <jni.h>
#include <tbb/tbb.h>

#include "algorithms/covariance/covariance_types.h"
#include "java_callback.h"

namespace daal
{
namespace algorithms
{
namespace covariance
{

using namespace daal::data_management;
using namespace daal::services;

/*
 * \brief Class that specifies the default method for partial results initialization
 */
struct JavaPartialResultInit : public PartialResultsInitIface, public JavaCallback
{
    JavaPartialResultInit(JavaVM *_jvm, jobject _javaObject) : JavaCallback(_jvm, _javaObject)
    {}

    virtual ~JavaPartialResultInit()
    {}

    /*
     * Initialize partial results
     * \param[in]       input     Input parameters of the Covariance algorithm
     * \param[in,out]   pres      Partial results of the Covariance algorithm
     */
    virtual void operator()(const Input &input, services::SharedPtr<PartialResult> &pres)
    {
        ThreadLocalStorage tls = _tls.local();
        jint status = jvm->AttachCurrentThread((void **)(&tls.jniEnv), NULL);
        JNIEnv *env = tls.jniEnv;

        jobject javaInput = constructJavaObjectFromCppObject(env, (jlong)(&input),
                                                             "com/intel/daal/algorithms/covariance/Input");

        SerializationIfacePtr serializablePRes = services::staticPointerCast<SerializationIface, PartialResult>(pres);
        jobject javaPartialResult = constructJavaObjectFromCppObject(env, (jlong)(&serializablePRes),
                                                                     "com/intel/daal/algorithms/covariance/PartialResult");

        jclass javaObjectClass = env->GetObjectClass(javaObject);
        if (javaObjectClass == NULL)
        {
            env->ThrowNew(env->FindClass("java/lang/Exception"), "Couldn't find class of this java object");
        }
        jmethodID initializeMethodID
            = env->GetMethodID(javaObjectClass, "initialize",
                               "(Lcom/intel/daal/algorithms/covariance/Input;Lcom/intel/daal/algorithms/covariance/PartialResult;)V");
        if (initializeMethodID == NULL)
        {
            env->ThrowNew(env->FindClass("java/lang/Exception"), "Couldn't find cInitialize method");
        }

        jlong partialResultAddr = env->CallLongMethod(javaObject, initializeMethodID, javaInput, javaPartialResult);
        if(!tls.is_main_thread)
        {
            status = jvm->DetachCurrentThread();
        }
        _tls.local() = tls;
    }
};

} // namespace daal::algorithms::covariance
} // namespace daal::algorithms
} // namespace daal

#endif
