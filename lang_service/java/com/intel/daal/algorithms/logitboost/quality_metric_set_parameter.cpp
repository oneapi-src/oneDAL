/* file: quality_metric_set_parameter.cpp */
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
#include "common_helpers.h"
#include "com_intel_daal_algorithms_logitboost_quality_metric_set_QualityMetricSetParameter.h"

using namespace daal::algorithms::logitboost::quality_metric_set;

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logitboost_quality_1metric_1set_QualityMetricSetParameter_cSetNClasses(JNIEnv *, jobject,
                                                                                                                             jlong self,
                                                                                                                             jlong nClasses)
{
    daal::unpack<Parameter>(self).nClasses = (size_t)nClasses;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logitboost_quality_1metric_1set_QualityMetricSetParameter_cGetNClasses(JNIEnv *, jobject,
                                                                                                                              jlong self)
{
    return (jlong)(daal::unpack<Parameter>(self).nClasses);
}
