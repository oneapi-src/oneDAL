/* file: init_input.cpp */
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

#include "daal.h"
#include "em_gmm/init/JInitInput.h"
#include "em_gmm/init/JInitInputId.h"
#include "common_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::em_gmm::init;

#define Data   com_intel_daal_algorithms_em_gmm_init_InitInputId_Data

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitInput
 * Method:    cInit
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitInput_cInit
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jint cmode)
{
    return jniBatch<em_gmm::init::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitInput
 * Method:    cSetInput
 * Signature: (JIJ)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitInput_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<em_gmm::init::Input>::set<em_gmm::init::InputId, NumericTable>(inputAddr, id, ntAddr);
    return (jint)0;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitInput
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitInput_cGetInputTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<em_gmm::init::Input>::get<em_gmm::init::InputId, NumericTable>(inputAddr, id);
}
