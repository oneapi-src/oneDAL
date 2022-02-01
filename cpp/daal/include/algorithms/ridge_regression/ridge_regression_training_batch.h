/* file: ridge_regression_training_batch.h */
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
//  Implementation of the interface for ridge regression model-based training in the batch processing mode
//--
*/

#ifndef __RIDGE_REGRESSION_TRAINING_BATCH_H__
#define __RIDGE_REGRESSION_TRAINING_BATCH_H__

#include "algorithms/algorithm.h"
#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "algorithms/ridge_regression/ridge_regression_training_types.h"
#include "algorithms/ridge_regression/ridge_regression_model.h"
#include "algorithms/linear_model/linear_model_training_batch.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
namespace interface1
{
/**
 * @defgroup ridge_regression_batch Batch
 * @ingroup ridge_regression_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__BATCHCONTAINER"></a>
 * \brief Class containing methods for normal equations ridge regression model-based training using algorithmFPType precision arithmetic
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for ridge regression model-based training with a specified environment in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~BatchContainer();

    /**
     * Computes the result of ridge regression model-based training in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__BATCH"></a>
 * \brief Provides methods for ridge regression model-based training in the batch processing mode
 * <!-- \n<a href="DAAL-REF-RIDGEREGRESSION-ALGORITHM">Ridge regression algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for ridge regression model-based training, double or float
 * \tparam method           Ridge regression training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref ridge_regression::interface1::Model "ridge_regression::Model" class
 *      - \ref ridge_regression::interface1::ModelNormEq "ridge_regression::ModelNormEq" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = normEqDense>
class DAAL_EXPORT Batch : public linear_model::training::Batch
{
public:
    typedef algorithms::ridge_regression::training::Input InputType;
    typedef algorithms::ridge_regression::TrainParameter ParameterType;
    typedef algorithms::ridge_regression::training::Result ResultType;

    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< %Training \ref interface1::Parameter "parameters" */

    /** Default constructor */
    Batch() { initialize(); }

    /**
     * Constructs a ridge regression training algorithm by copying input objects
     * and parameters of another ridge regression training algorithm in the batch processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other) : input(other.input), parameter(other.parameter) { initialize(); }

    ~Batch() {}

    virtual regression::training::Input * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the result of ridge regression model-based training
     * \return Structure that contains the result of ridge regression model-based training
     */
    ResultPtr getResult() { return ResultType::cast(_result); }

    /* Resets the results of the regression model-based training
     * \return Status of the operation
     */
    virtual services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult)
        _res = NULL;
        return services::Status();
    }
    /**
     * Returns a pointer to a newly allocated ridge regression training algorithm
     * with a copy of the input objects and parameters for this ridge regression training algorithm
     * in the batch processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = getResult()->template allocate<algorithmFPType>(&input, &parameter, method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }

private:
    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface1

using interface1::BatchContainer;
using interface1::Batch;

} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
