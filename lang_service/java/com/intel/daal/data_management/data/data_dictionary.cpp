/* file: data_dictionary.cpp */
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
#include <string>

#include "JDataDictionary.h"
#include "data_dictionary.h"
#include "numeric_table.h"
#include "common_helpers_functions.h"

using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_DataDictionary
 * Method:    init
 * Signature:(I)V
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_DataDictionary_init
(JNIEnv *env, jobject thisObj, jint nFeatures, jint featuresEqual)
{
    using namespace daal;

    // Create C++ object of the class NumericTableDictionary
    NumericTableDictionary* dict = new NumericTableDictionary((size_t)nFeatures, (DictionaryIface::FeaturesEqual)featuresEqual);
    return (jlong)new NumericTableDictionaryPtr(dict);
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
    NumericTableDictionary *dict = ((NumericTableDictionaryPtr*)cObject)->get();
    dict->resetDictionary();
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
    NumericTableDictionary *dict = ((NumericTableDictionaryPtr*)cObject)->get();
    DAAL_CHECK_THROW(dict->setFeature(*((services::SharedPtr<NumericTableFeature> *)dfAddr)->get(), (size_t)idx));
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
    NumericTableDictionary *dict = ((NumericTableDictionaryPtr*)cObject)->get();

    dict->setAllFeatures(*((services::SharedPtr<NumericTableFeature> *)dfAddr)->get());
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
    NumericTableDictionary *dict = ((NumericTableDictionaryPtr*)cObject)->get();

    dict->setNumberOfFeatures((size_t)nFeatures);
}

/*
 * Class:     com_intel_daal_data_1management_data_DataDictionary
 * Method:    cGetCDataDictionaryFeaturesEqual
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_DataDictionary_cGetCDataDictionaryFeaturesEqual
(JNIEnv *env, jobject thisObj, jlong cObject)
{
    using namespace daal;
    NumericTableDictionary *dict = ((NumericTableDictionaryPtr *)cObject)->get();
    return (jint)(dict->getFeaturesEqual());
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
    NumericTableDictionary *dict = ((NumericTableDictionaryPtr *)cObject)->get();
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
    NumericTableDictionary *dict = ((NumericTableDictionaryPtr*)cObject)->get();

    return (int)(dict->getNumberOfFeatures());
}
