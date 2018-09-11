/* file: java_batch_container.h */
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

    virtual services::Status compute()
    {
        return daal::services::JavaBatchContainerService::compute(
            "Lcom/intel/daal/algorithms/covariance/Result;",
            "Lcom/intel/daal/algorithms/covariance/Input;",
            "Lcom/intel/daal/algorithms/covariance/Parameter;"
            );
    }

    void setJavaResult(ResultPtr result)
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
