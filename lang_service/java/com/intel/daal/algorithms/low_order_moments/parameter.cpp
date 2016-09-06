/* file: parameter.cpp */
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
#include "low_order_moments/JParameter.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::low_order_moments;

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_Parameter
 * Method:    cSetInitializationProcedure
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_Parameter_cSetInitializationProcedure
(JNIEnv *env, jobject thisObj, jlong parAddr, jlong initAddr)
{
    using namespace daal::algorithms;
    low_order_moments::Parameter *parameterAddr = (low_order_moments::Parameter *)parAddr;

    parameterAddr->initializationProcedure = *((SharedPtr<low_order_moments::PartialResultsInitIface> *)initAddr);
}
