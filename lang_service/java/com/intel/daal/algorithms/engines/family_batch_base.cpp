/* file: family_batch_base.cpp */
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

#include <jni.h>
#include "engines/JFamilyBatchBase.h"

#include "daal.h"

using namespace daal;
using namespace daal::services;
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_engines_FamilyBatchBase
 * Method:    cAdd
 * Signature: (J)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_engines_FamilyBatchBase_cAdd
  (JNIEnv *env, jobject thisObj, jlong algAddr, jlong numberOfStreams)
{
    (staticPointerCast<engines::FamilyBatchBase, AlgorithmIface>(
        *((SharedPtr<AlgorithmIface> *)algAddr)))->add(numberOfStreams);
}

/*
 * Class:     com_intel_daal_algorithms_engines_FamilyBatchBase
 * Method:    cGetNumberOfStreams
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_engines_FamilyBatchBase_cGetNumberOfStreams
  (JNIEnv *env, jobject thisObj, jlong algAddr)
{
    return (staticPointerCast<engines::FamilyBatchBase, AlgorithmIface>(
        *((SharedPtr<AlgorithmIface> *)algAddr)))->getNumberOfStreams();
}

/*
 * Class:     com_intel_daal_algorithms_engines_FamilyBatchBase
 * Method:    cGetMaxNumberOfStreams
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_engines_FamilyBatchBase_cGetMaxNumberOfStreams
  (JNIEnv *env, jobject thisObj, jlong algAddr)
{
    return (staticPointerCast<engines::FamilyBatchBase, AlgorithmIface>(
        *((SharedPtr<AlgorithmIface> *)algAddr)))->getMaxNumberOfStreams();
}
