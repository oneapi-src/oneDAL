/* file: parameter.cpp */
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

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>/* Header for class com_intel_daal_algorithms_association_rules_Batch */

#include "daal.h"
#include "com_intel_daal_algorithms_association_rules_Parameter.h"

using namespace daal;
using namespace daal::algorithms;

#include "associationrules_types.i"

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cSetMinSupport(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                                  jdouble val)
{
    (*(association_rules::Parameter *)parAddr).minSupport = val;
}

JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cGetMinSupport(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    return (jdouble)(*(association_rules::Parameter *)parAddr).minSupport;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cSetMinConfidence(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                                     jdouble val)
{
    (*(association_rules::Parameter *)parAddr).minConfidence = val;
}

JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cGetMinConfidence(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    return (jdouble)(*(association_rules::Parameter *)parAddr).minConfidence;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cSetNUniqueItems(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                                    jlong val)
{
    (*(association_rules::Parameter *)parAddr).nUniqueItems = val;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cGetNUniqueItems(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    return (jlong)(*(association_rules::Parameter *)parAddr).nUniqueItems;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cSetNTransactions(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                                     jlong val)
{
    (*(association_rules::Parameter *)parAddr).nTransactions = val;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cGetNTransactions(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    return (jlong)(*(association_rules::Parameter *)parAddr).nTransactions;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cSetDiscoverRules(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                                     jboolean flag)
{
    (*(association_rules::Parameter *)parAddr).discoverRules = flag;
}

JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cGetDiscoverRules(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    return (jboolean)(*(association_rules::Parameter *)parAddr).discoverRules;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cSetMinUniqueItemsetSize(JNIEnv * env, jobject thisObj,
                                                                                                            jlong parAddr, jlong val)
{
    (*(association_rules::Parameter *)parAddr).minItemsetSize = val;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cGetMinUniqueItemsetSize(JNIEnv * env, jobject thisObj,
                                                                                                             jlong parAddr)
{
    return (jlong)(*(association_rules::Parameter *)parAddr).minItemsetSize;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cSetMaxItemsetSize(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                                      jlong val)
{
    (*(association_rules::Parameter *)parAddr).maxItemsetSize = val;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cGetMaxItemsetSize(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    return (jlong)(*(association_rules::Parameter *)parAddr).maxItemsetSize;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cSetItemsetsOrder(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                                     jint id)
{
    if (id == ItemsetsUnsorted)
    {
        (*(association_rules::Parameter *)parAddr).itemsetsOrder = association_rules::itemsetsUnsorted;
    }
    else if (id == ItemsetsSortedBySupport)
    {
        (*(association_rules::Parameter *)parAddr).itemsetsOrder = association_rules::itemsetsSortedBySupport;
    }
}

JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cGetItemsetsOrder(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    return (jint)(*(association_rules::Parameter *)parAddr).itemsetsOrder;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cSetRulesOrder(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                                  jint id)
{
    if (id == RulesUnsorted)
    {
        (*(association_rules::Parameter *)parAddr).rulesOrder = association_rules::rulesUnsorted;
    }
    else if (id == RulesSortedByConfidence)
    {
        (*(association_rules::Parameter *)parAddr).rulesOrder = association_rules::rulesSortedByConfidence;
    }
}

JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_association_1rules_Parameter_cGetRulesOrder(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    return (jint)(*(association_rules::Parameter *)parAddr).rulesOrder;
}
