/* file: data_dictionary.cpp */
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
#include <string>

#include "JDataDictionary.h"
#include "data_dictionary.h"
#include "numeric_table.h"

using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_DataDictionary
 * Method:    init
 * Signature:(I)V
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_DataDictionary_init
(JNIEnv *env, jobject thisObj, jint nFeatures, jboolean featuresEqual)
{
    using namespace daal;

    // Create C++ object of the class NumericTableDictionary
    services::SharedPtr<NumericTableDictionary> *dict =
        new services::SharedPtr<NumericTableDictionary>(new NumericTableDictionary((size_t)nFeatures, (bool)featuresEqual));

    if((*dict)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), (*dict)->getErrors()->getDescription());
    }

    return (jlong)dict;
}

/*
 * Class:     com_intel_daal_data_1management_data_DataDictionary
 * Method:    cResetDictionary
 * Signature:()V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_DataDictionary_cResetDictionary
(JNIEnv *env, jobject thisObj, jlong cObject)
{
    using namespace daal;

    // Get a class reference for Java NumericTableDictionary
    NumericTableDictionary *dict = ((services::SharedPtr<NumericTableDictionary>*)cObject)->get();
    dict->resetDictionary();

    if(dict->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), dict->getErrors()->getDescription());
    }
}

/*
 * Class:     com_intel_daal_data_1management_data_DataDictionary
 * Method:    cSetFeature
 * Signature:(JI)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_DataDictionary_cSetFeature
(JNIEnv *env, jobject thisObj, jlong cObject, jlong dfAddr, jint idx)
{
    using namespace daal;

    // Get a class reference for Java NumericTableDictionary
    NumericTableDictionary *dict = ((services::SharedPtr<NumericTableDictionary>*)cObject)->get();

    dict->setFeature(*((services::SharedPtr<NumericTableFeature> *)dfAddr)->get(), (size_t)idx);

    if(dict->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), dict->getErrors()->getDescription());
    }
}


/*
 * Class:     com_intel_daal_data_1management_data_DataDictionary
 * Method:    cSetAllFeatures
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_DataDictionary_cSetAllFeatures
(JNIEnv *env, jobject thisObj, jlong cObject, jlong dfAddr)
{
    using namespace daal;

    // Get a class reference for Java NumericTableDictionary
    NumericTableDictionary *dict = ((services::SharedPtr<NumericTableDictionary>*)cObject)->get();

    dict->setAllFeatures(*((services::SharedPtr<NumericTableFeature> *)dfAddr)->get());

    if(dict->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), dict->getErrors()->getDescription());
    }
}


/*
 * Class:     com_intel_daal_data_1management_data_DataDictionary
 * Method:    cSetNumberOfFeatures
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_DataDictionary_cSetNumberOfFeatures
(JNIEnv *env, jobject thisObj, jlong cObject, jlong nFeatures)
{
    using namespace daal;

    // Get a class reference for Java NumericTableDictionary
    NumericTableDictionary *dict = ((services::SharedPtr<NumericTableDictionary>*)cObject)->get();

    dict->setNumberOfFeatures((size_t)nFeatures);

    if(dict->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), dict->getErrors()->getDescription());
    }
}

/*
 * Class:     com_intel_daal_data_1management_data_DataDictionary
 * Method:    cGetCDataDictionaryFeaturesEqual
 * Signature:(J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_data_1management_data_DataDictionary_cGetCDataDictionaryFeaturesEqual
(JNIEnv *env, jobject thisObj, jlong cObject)
{
    using namespace daal;
    NumericTableDictionary *dict = ((services::SharedPtr<NumericTableDictionary> *)cObject)->get();
    return (jboolean)(dict->getFeaturesEqual());
}

/*
 * Class:     com_intel_daal_data_1management_data_DataDictionary
 * Method:    cGetIndexType
 * Signature:(JI)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_DataDictionary_cGetIndexType
(JNIEnv *env, jobject thisObj, jlong cObject, jint idx)
{
    using namespace daal;
    NumericTableDictionary *dict = ((services::SharedPtr<NumericTableDictionary> *)cObject)->get();
    return (jint)((*dict)[idx].indexType);
}

/*
 * Class:     com_intel_daal_data_management_data_DataDictionary
 * Method:    cGetNumberOfFeatures
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_DataDictionary_cGetNumberOfFeatures
  (JNIEnv *env, jobject thosObj, jlong cObject)
{
    using namespace daal;

    // Get a class reference for Java NumericTableDictionary
    NumericTableDictionary *dict = ((services::SharedPtr<NumericTableDictionary>*)cObject)->get();

    return (int)(dict->getNumberOfFeatures());
}
