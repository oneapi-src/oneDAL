/* file: input.cpp */
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
#include "daal.h"
#include "em_gmm/JInput.h"
#include "em_gmm/JInputId.h"
#include "em_gmm/JMethod.h"
#include "common_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::em_gmm;

#define DefaultDense com_intel_daal_algorithms_em_gmm_Method_defaultDenseValue
#define Data         com_intel_daal_algorithms_em_gmm_InputId_DefaultInputId

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Input
 * Method:    cInit
 * Signature: (JIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Input_cInit
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jint cmode)
{
    return jniBatch<em_gmm::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Input
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Input_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<em_gmm::Input>::set<em_gmm::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Input
 * Method:    cSetInputInputValues
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Input_cSetInputInputValues
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong initResultAddr)
{
    jniInput<em_gmm::Input>::set<em_gmm::InputValuesId, em_gmm::init::Result>(inputAddr, id, initResultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Input
 * Method:    cSetInputCovariances
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Input_cSetInputCovariances
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong initCovariancesCollectionAddr)
{
    jniInput<em_gmm::Input>::set<em_gmm::InputCovariancesId, DataCollection>(inputAddr, id, initCovariancesCollectionAddr);
}


/*
 * Class:     com_intel_daal_algorithms_em_gmm_Input
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Input_cGetInputTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<em_gmm::Input>::get<em_gmm::InputId, NumericTable>(inputAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Input
 * Method:    cGetInputCovariancesDataCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Input_cGetInputCovariancesDataCollection
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<em_gmm::Input>::get<em_gmm::InputCovariancesId, DataCollection>(inputAddr, id);
}
