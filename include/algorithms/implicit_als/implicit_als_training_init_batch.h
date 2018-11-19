/* file: implicit_als_training_init_batch.h */
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
    services::Status compute() DAAL_C11_OVERRIDE;
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
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Training<batch>
{
public:
    typedef algorithms::implicit_als::training::init::Input     InputType;
    typedef algorithms::implicit_als::training::init::Parameter ParameterType;
    typedef algorithms::implicit_als::training::init::Result    ResultType;

    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< %Algorithm parameter */

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
    Batch(const Batch<algorithmFPType, method> &other) : input(other.input), parameter(other.parameter)
    {
        initialize();
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
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store the results of the implicit ALS initialization algorithm
     * \param[in] res  Structure to store the results of the implicit ALS initialization algorithm
     */
    services::Status setResult(const ResultPtr& res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res = _result.get();
        return services::Status();
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

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, (int) method);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }

private:
    ResultPtr _result;
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
