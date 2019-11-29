/* file: environment.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "com_intel_daal_services_Environment.h"
#include "daal.h"

using namespace daal::services;

/*
 * Class:     com_intel_daal_services_Environment
 * Method:    cGetCpuId
 * Signature: (I)V
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_services_Environment_cGetCpuId(JNIEnv *, jclass, jint enable)
{
    return Environment::getInstance()->getCpuId(enable);
}

/*
 * Class:     com_intel_daal_services_Environment
 * Method:    cSetCpuId
 * Signature: (I)V
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_services_Environment_cSetCpuId(JNIEnv *, jclass, jint cpuid)
{
    return Environment::getInstance()->setCpuId(cpuid);
}

/*
 * Class:     com_intel_daal_services_Environment
 * Method:    cEnableInstructionsSet
 * Signature: (I)V
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_services_Environment_cEnableInstructionsSet(JNIEnv *, jclass, jint enable)
{
    return Environment::getInstance()->enableInstructionsSet(enable);
}

/*
 * Class:     com_intel_daal_services_Environment
 * Method:    cSetNumberOfThreads
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_services_Environment_cSetNumberOfThreads(JNIEnv *, jclass, jint numThreads)
{
    Environment::getInstance()->setNumberOfThreads(numThreads);
}

/*
 * Class:     com_intel_daal_services_Environment
 * Method:    cGetNumberOfThreads
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_services_Environment_cGetNumberOfThreads(JNIEnv *, jclass)
{
    return Environment::getInstance()->getNumberOfThreads();
}

/*
 * Class:     com_intel_daal_services_Environment
 * Method:    cEnableThreadPinning
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_services_Environment_cEnableThreadPinning(JNIEnv *, jclass, jboolean enableThreadPinningFlag)
{
    return Environment::getInstance()->enableThreadPinning(enableThreadPinningFlag);
}
