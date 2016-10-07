/* file: factory.cpp */
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

#include "JFactory.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::services;

/*
 * Class:     com_intel_daal_data_management_data_Factory
 * Method:    cGetSerializationTag
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_Factory_cGetSerializationTag
  (JNIEnv *env, jobject thisObj, jlong serializableAddr)
{
    SerializationIfacePtr *object = (SerializationIfacePtr *)serializableAddr;
    int tag = (*object)->getSerializationTag();
    return (jint)tag;
}
