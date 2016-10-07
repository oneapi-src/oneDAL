/* file: initialization_procedure_iface.cpp */
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

#include <jni.h>

#include "daal.h"
#include "initialization_procedure.h"

#include "covariance/JInitializationProcedureIface.h"

using namespace daal;
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_covariance_InitializationProcedureIface
 * Method:    newJavaInitializationProcedure
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_InitializationProcedureIface_cNewJavaInitializationProcedure
(JNIEnv *env, jobject thisObj)
{
    JavaVM *jvm;
    // Get pointer to the Java VM interface function table
    jint status = env->GetJavaVM(&jvm);
    if(status != 0)
    {
        return 0;
    }

    covariance::PartialResultsInitIface *initializationProcedure =
        new covariance::JavaPartialResultInit(jvm, thisObj);

    services::SharedPtr<covariance::PartialResultsInitIface> *initializationProcedureShPtr =
        new services::SharedPtr<covariance::PartialResultsInitIface>(initializationProcedure);

    return (jlong)initializationProcedureShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_covariance_InitializationProcedureIface
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_InitializationProcedureIface_cDispose
(JNIEnv *env, jobject thisObj, jlong initAddr)
{
    services::SharedPtr<covariance::PartialResultsInitIface> *initializationProcedureShPtr =
        (services::SharedPtr<covariance::PartialResultsInitIface> *)initAddr;

    delete initializationProcedureShPtr;
}
