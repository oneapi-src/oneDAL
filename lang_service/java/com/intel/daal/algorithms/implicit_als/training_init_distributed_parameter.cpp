/* file: training_init_distributed_parameter.cpp */
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

#include "daal.h"

#include "com_intel_daal_algorithms_implicit_als_training_init_InitDistributedParameter.h"

using namespace daal;
using namespace daal::algorithms::implicit_als::training::init;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitParameter
 * Method:    cSetPartition
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedParameter_cSetPartition(JNIEnv *, jobject,
                                                                                                                         jlong parAddr, jlong ntAddr)
{
    NumericTablePtr nt                           = NumericTable::cast(*(SerializationIfacePtr *)ntAddr);
    ((DistributedParameter *)parAddr)->partition = nt;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitParameter
 * Method:    cGetPartition
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL ava_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedParameter_cGetPartition(JNIEnv *, jobject,
                                                                                                                         jlong parAddr)
{
    return (jlong)(new SerializationIfacePtr(((DistributedParameter *)parAddr)->partition));
}
