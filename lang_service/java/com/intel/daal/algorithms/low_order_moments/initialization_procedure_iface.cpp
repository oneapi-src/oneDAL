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

#include "low_order_moments/JInitializationProcedureIface.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_InitializationProcedureIface
 * Method:    cNewJavaInitializationProcedure
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_InitializationProcedureIface_cNewJavaInitializationProcedure
(JNIEnv *env, jobject thisObj)
{
    JavaVM *jvm;
    // Get pointer to the Java VM interface function table
    jint status = env->GetJavaVM(&jvm);
    if(status != 0)
    {
        return 0;
    }

    low_order_moments::PartialResultsInitIface *initializationProcedure =
        new low_order_moments::JavaPartialResultInit(jvm, thisObj);

    SharedPtr<low_order_moments::PartialResultsInitIface> *initializationProcedureShPtr =
        new SharedPtr<low_order_moments::PartialResultsInitIface>(initializationProcedure);

    return (jlong)initializationProcedureShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_InitializationProcedureIface
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_InitializationProcedureIface_cDispose
(JNIEnv *env, jobject thisObj, jlong initAddr)
{
    SharedPtr<low_order_moments::PartialResultsInitIface> *initializationProcedureShPtr =
        (SharedPtr<low_order_moments::PartialResultsInitIface> *)initAddr;

    delete initializationProcedureShPtr;
}
