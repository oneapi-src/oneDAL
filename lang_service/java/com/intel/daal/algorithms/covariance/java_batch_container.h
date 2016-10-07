/* file: java_batch_container.h */
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

/*
//++
//  Implementation of the class that connects Covariance Java Batch
//  to C++ algorithm
//--
*/
#ifndef __JAVA_BATCH_CONTAINER_COVARIANCE_H__
#define __JAVA_BATCH_CONTAINER_COVARIANCE_H__

#include <jni.h>

#include "algorithms/covariance/covariance_types.h"
#include "algorithms/covariance/covariance_batch.h"
#include "java_callback.h"
#include "java_batch_container_service.h"

using namespace daal::algorithms::covariance;

namespace daal
{
namespace algorithms
{
namespace covariance
{

class JavaBatchContainer : public daal::services::JavaBatchContainerService
{
public:
    JavaBatchContainer(JavaVM *_jvm, jobject _javaObject) : JavaBatchContainerService(_jvm, _javaObject) {};
    JavaBatchContainer(const JavaBatchContainer &other) : JavaBatchContainerService(other) {}
    virtual ~JavaBatchContainer() {}

    virtual void compute()
    {
        daal::services::JavaBatchContainerService::compute(
            "Lcom/intel/daal/algorithms/covariance/Result;",
            "Lcom/intel/daal/algorithms/covariance/Input;",
            "Lcom/intel/daal/algorithms/covariance/Parameter;"
            );
    }

    void setJavaResult(services::SharedPtr<Result> result)
    {
        _result = result;
    };

    virtual JavaBatchContainer * cloneImpl()
    {
        return new JavaBatchContainer(*this);
    }
};

} // namespace daal::algorithms::covariance
} // namespace daal::algorithms
} // namespace daal

#endif
