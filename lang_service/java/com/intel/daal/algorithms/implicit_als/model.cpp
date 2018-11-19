/* file: model.cpp */
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

#include "implicit_als/JModel.h"

using namespace daal;
using namespace daal::algorithms::implicit_als;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Model
 * Method:    cGetUsersFactors
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_Model_cGetUsersFactors
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    NumericTablePtr *nt = new NumericTablePtr();
    algorithms::implicit_als::ModelPtr res = *(algorithms::implicit_als::ModelPtr *)modAddr;

    *nt = res->getUsersFactors();

    return (jlong)nt;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Model
 * Method:    cGetItemsFactors
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_Model_cGetItemsFactors
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    NumericTablePtr *nt = new NumericTablePtr();
    algorithms::implicit_als::ModelPtr res = *(algorithms::implicit_als::ModelPtr *)modAddr;

    *nt = res->getItemsFactors();

    return (jlong)nt;
}
