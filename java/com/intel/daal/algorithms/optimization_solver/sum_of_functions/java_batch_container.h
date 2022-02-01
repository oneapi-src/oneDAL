/* file: java_batch_container.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of the class that connects Java Batch
//  to C++ algorithm
//--
*/
#ifndef __JAVA_BATCH_CONTAINER_SUM_OF_FUNCTIONS_H__
#define __JAVA_BATCH_CONTAINER_SUM_OF_FUNCTIONS_H__

#include <jni.h>

#include "algorithms/optimization_solver/objective_function/sum_of_functions_types.h"
#include "algorithms/optimization_solver/objective_function/sum_of_functions_batch.h"
#include "com/intel/daal/java_callback.h"
#include "com/intel/daal/java_batch_container_service.h"

using namespace daal::algorithms::optimization_solver;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sum_of_functions
{
class JavaBatchContainer : public daal::services::JavaBatchContainerService
{
public:
    JavaBatchContainer(JavaVM * _jvm, jobject _javaObject) : JavaBatchContainerService(_jvm, _javaObject) {};
    JavaBatchContainer(const JavaBatchContainer & other) : JavaBatchContainerService(other) {}

    virtual ~JavaBatchContainer() {}

    virtual services::Status compute()
    {
        return daal::services::JavaBatchContainerService::compute("Lcom/intel/daal/algorithms/optimization_solver/objective_function/Result;");
    }

    void setJavaResult(objective_function::ResultPtr result) { _result = result; };

    virtual JavaBatchContainer * cloneImpl() { return new JavaBatchContainer(*this); }
};

} // namespace sum_of_functions
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
