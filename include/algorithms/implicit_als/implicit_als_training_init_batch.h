/* file: implicit_als_training_init_batch.h */
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
//  Implementation of the interface for the implicit ALS initialization algorithm
//  in the batch processing mode
//--
*/

#ifndef __IMPLICIT_ALS_TRAINING_INIT_BATCH_H__
#define __IMPLICIT_ALS_TRAINING_INIT_BATCH_H__

#include "algorithms/algorithm.h"
#include "algorithms/implicit_als/implicit_als_training_init_types.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{

namespace interface1
{
/**
 * @defgroup implicit_als_init_batch Batch
 * @ingroup implicit_als_init
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the implicit ALS initialization algorithm
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public TrainingContainerIface<batch>
{
public:
     /**
     * Constructs a container for the implicit ALS initialization algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes initial values for implicit ALS model-based training in the batch processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__BATCH"></a>
 * \brief Algorithm class for initializing the implicit ALS model
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the implicit ALS initialization algorithm, double or float
 * \tparam method           Implicit ALS initialization method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method    Implicit ALS initialization method
 *      - \ref InputId   Identifiers of input objects for the implicit ALS initialization algorithm
 *      - \ref ResultId  Identifiers of the results of the implicit ALS initialization algorithm
 *
 * \par References
 *      - \ref Parameter class
 *      - \ref Input class
 *      - \ref Result class
 */
template<typename algorithmFPType = double, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Training<batch>
{
public:
    Input input;         /*!< %Input data structure */
    Parameter parameter; /*!< %Algorithm parameter */

    /** Default constructor */
    Batch()
    {
        initialize();
    }

    /**
     * Constructs an algorithm for initializing the implicit ALS model by copying input objects and parameters
     * of another algorithm for initializing the implicit ALS model
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other)
    {
        initialize();
        input.set(data,       other.input.get(data));
        parameter = other.parameter;
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains the results of the implicit ALS initialization algorithm
     * \return Structure that contains the results of the implicit ALS initialization algorithm
     */
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store the results of the implicit ALS initialization algorithm
     * \param[in] res  Structure to store the results of the implicit ALS initialization algorithm
     */
    void setResult(const services::SharedPtr<Result>& res)
    {
        DAAL_CHECK(res, ErrorNullResult)
        _result = res;
        _res = _result.get();
    }

    /**
     * Returns a pointer to the newly allocated algorithm for initializing the implicit ALS model with a copy of input objects
     * of this algorithm for initializing the implicit ALS model
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    virtual void allocateResult() DAAL_C11_OVERRIDE
    {
        _result->allocate<algorithmFPType>(&input, &parameter, (int) method);
        _res = _result.get();
    }

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _result = services::SharedPtr<Result>(new Result());
    }

private:
    services::SharedPtr<Result> _result;
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

}
}
}
}
}

#endif
