/* file: distributed_partial_result_step6.cpp */
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
#include "com_intel_daal_algorithms_gbt_regression_training_DistributedPartialResultStep6.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::gbt::regression::training;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_training_DistributedPartialResultStep6_cNewDistributedPartialResultStep6
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<DistributedPartialResultStep6>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_gbt_regression_training_DistributedPartialResultStep6
 * Method:    cGetModel
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_training_DistributedPartialResultStep6_cGetModel
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    ModelPtr *m = new ModelPtr();
    DistributedPartialResultStep6Ptr res = services::staticPointerCast<DistributedPartialResultStep6, SerializationIface>(*((SerializationIfacePtr *)resAddr));
    *m = res->get(partialModel);
    return (jlong)m;
}
