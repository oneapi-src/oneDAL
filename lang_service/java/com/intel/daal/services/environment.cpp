/* file: environment.cpp */
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

#include <jni.h>

#include "JEnvironment.h"
#include "daal.h"

using namespace daal::services;

/*
 * Class:     com_intel_daal_services_Environment
 * Method:    cGetCpuId
 * Signature: (I)V
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_services_Environment_cGetCpuId
(JNIEnv *, jclass, jint enable)
{
    return Environment::getInstance()->getCpuId(enable);
}

/*
 * Class:     com_intel_daal_services_Environment
 * Method:    cSetCpuId
 * Signature: (I)V
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_services_Environment_cSetCpuId
(JNIEnv *, jclass, jint cpuid)
{
    return Environment::getInstance()->setCpuId(cpuid);
}

/*
 * Class:     com_intel_daal_services_Environment
 * Method:    cEnableInstructionsSet
 * Signature: (I)V
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_services_Environment_cEnableInstructionsSet
(JNIEnv *, jclass, jint enable)
{
    return Environment::getInstance()->enableInstructionsSet(enable);
}

/*
 * Class:     com_intel_daal_services_Environment
 * Method:    cSetNumberOfThreads
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_services_Environment_cSetNumberOfThreads
(JNIEnv *, jclass, jint numThreads)
{
    Environment::getInstance()->setNumberOfThreads(numThreads);
}

/*
 * Class:     com_intel_daal_services_Environment
 * Method:    cGetNumberOfThreads
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_services_Environment_cGetNumberOfThreads
(JNIEnv *, jclass)
{
    return Environment::getInstance()->getNumberOfThreads();
}

/*
 * Class:     com_intel_daal_services_Environment
 * Method:    cEnableThreadPinning
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_services_Environment_cEnableThreadPinning
  (JNIEnv *, jclass, jboolean enableThreadPinningFlag)
{
    return Environment::getInstance()->enableThreadPinning(enableThreadPinningFlag);
}
