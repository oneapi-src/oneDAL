/* file: java_batch_container.h */
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
#include "java_callback.h"
#include "java_batch_container_service.h"

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
    JavaBatchContainer(JavaVM *_jvm, jobject _javaObject) : JavaBatchContainerService(_jvm, _javaObject) {};
    JavaBatchContainer(const JavaBatchContainer &other) : JavaBatchContainerService(other) {}

    virtual ~JavaBatchContainer() {}

    virtual services::Status compute()
    {
        return daal::services::JavaBatchContainerService::compute(
            "Lcom/intel/daal/algorithms/optimization_solver/objective_function/Result;");
    }

    void setJavaResult(objective_function::ResultPtr result)
    {
        _result = result;
    };

    virtual JavaBatchContainer * cloneImpl()
    {
        return new JavaBatchContainer(*this);
    }
};

} // namespace daal::algorithms::optimization_solver::sum_of_functions
} // namespace daal::algorithms::optimization_solver
} // namespace daal::algorithms
} // namespace daal

#endif
