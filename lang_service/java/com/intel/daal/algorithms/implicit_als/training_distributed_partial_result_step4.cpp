/* file: training_distributed_partial_result_step4.cpp */
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

#include "implicit_als/training/JDistributedPartialResultStep4.h"

#include "implicit_als_training_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als;
using namespace daal::algorithms::implicit_als::training;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedPartialResultStep4
 * Method:    cNewDistributedPartialResultStep4
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedPartialResultStep4_cNewDistributedPartialResultStep4
  (JNIEnv *env, jobject thisObj)
{
    return jniArgument<DistributedPartialResultStep4>::newObj();
}


/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedPartialResultStep4
 * Method:    cGetPartialModel
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedPartialResultStep4_cGetPartialModel
  (JNIEnv *env, jobject thisObj, jlong partialResultAddr, jint id)
{
    return jniArgument<implicit_als::training::DistributedPartialResultStep4>::
        get<DistributedPartialResultStep4Id, PartialModel>(partialResultAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedPartialResultStep4
 * Method:    cSetPartialModel
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedPartialResultStep4_cSetPartialModel
  (JNIEnv *env, jobject thisObj, jlong partialResultAddr, jint id, jlong partialModelAddr)
{
    jniArgument<implicit_als::training::DistributedPartialResultStep4>::
        set<DistributedPartialResultStep4Id, PartialModel>(partialResultAddr, id, partialModelAddr);
}
