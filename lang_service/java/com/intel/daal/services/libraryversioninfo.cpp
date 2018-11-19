/* file: libraryversioninfo.cpp */
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
 * Method:    cGetBuildRev
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_intel_daal_services_LibraryVersionInfo_cGetBuildRev
(JNIEnv *env, jobject, jlong verinfo)
{
    const char *str = ((LibraryVersionInfo *)verinfo)->build_rev;

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
