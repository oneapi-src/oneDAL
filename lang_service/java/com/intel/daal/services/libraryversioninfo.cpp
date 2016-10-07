/* file: libraryversioninfo.cpp */
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

#include "JLibraryVersionInfo.h"
#include "daal.h"

using namespace daal::services;

/*
 * Class:     com_intel_daal_services_LibraryVersionInfo
 * Method:    cGetMajorVersion
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_services_LibraryVersionInfo_cGetMajorVersion
(JNIEnv *, jobject, jlong verinfo)
{
    return ((LibraryVersionInfo *)verinfo)->majorVersion;
}

/*
 * Class:     com_intel_daal_services_LibraryVersionInfo
 * Method:    cGetMinorVersion
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_services_LibraryVersionInfo_cGetMinorVersion
(JNIEnv *, jobject, jlong verinfo)
{
    return ((LibraryVersionInfo *)verinfo)->minorVersion;
}

/*
 * Class:     com_intel_daal_services_LibraryVersionInfo
 * Method:    cGetUpdateVersion
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_services_LibraryVersionInfo_cGetUpdateVersion
(JNIEnv *, jobject, jlong verinfo)
{
    return ((LibraryVersionInfo *)verinfo)->updateVersion;
}

/*
 * Class:     com_intel_daal_services_LibraryVersionInfo
 * Method:    cGetProductStatus
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_intel_daal_services_LibraryVersionInfo_cGetProductStatus
(JNIEnv *env, jobject, jlong verinfo)
{
    const char *str = ((LibraryVersionInfo *)verinfo)->productStatus;

    return env->NewStringUTF(str);
}

/*
 * Class:     com_intel_daal_services_LibraryVersionInfo
 * Method:    cGetBuild
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_intel_daal_services_LibraryVersionInfo_cGetBuild
(JNIEnv *env, jobject, jlong verinfo)
{
    const char *str = ((LibraryVersionInfo *)verinfo)->build;

    return env->NewStringUTF(str);
}

/*
 * Class:     com_intel_daal_services_LibraryVersionInfo
 * Method:    cGetName
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_intel_daal_services_LibraryVersionInfo_cGetName
(JNIEnv *env, jobject, jlong verinfo)
{
    const char *str = ((LibraryVersionInfo *)verinfo)->name;

    return env->NewStringUTF(str);
}

/*
 * Class:     com_intel_daal_services_LibraryVersionInfo
 * Method:    cGetProcessor
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_intel_daal_services_LibraryVersionInfo_cGetProcessor
(JNIEnv *env, jobject, jlong verinfo)
{
    const char *str = ((LibraryVersionInfo *)verinfo)->processor;

    return env->NewStringUTF(str);
}

/*
 * Class:     com_intel_daal_services_LibraryVersionInfo
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_services_LibraryVersionInfo_cInit
(JNIEnv *, jobject)
{
    jlong verinfo = 0;
    verinfo = (jlong)(new LibraryVersionInfo());
    return verinfo;
}
