/* file: onlineparameter.cpp */
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
#include "covariance/JOnlineParameter.h"

#include "covariance_types.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::covariance;

/*
 * Class:     com_intel_daal_algorithms_covariance_Parameter
 * Method:    cSetCParameterObject
 * Signature: (JJIIII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_OnlineParameter_cSetCParameterObject
(JNIEnv *env, jobject thisObj, jlong parameterAddr, jlong algAddr)
{
    using namespace daal::services;
    using namespace daal::algorithms;
    using namespace daal::data_management;
    staticPointerCast<covariance::OnlineImpl, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->parameter =
        *((covariance::OnlineParameter *)parameterAddr);
}
