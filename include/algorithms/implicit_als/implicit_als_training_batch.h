/* file: implicit_als_training_batch.h */
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
//  Implementation of the interface for implicit ALS model-based training in the
//  batch processing mode
//--
*/

#ifndef __IMPLICIT_ALS_TRAINING_BATCH_H__
#define __IMPLICIT_ALS_TRAINING_BATCH_H__

#include "algorithms/algorithm.h"
#include "algorithms/implicit_als/implicit_als_training_types.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{

namespace interface1
{
/**
 * @defgroup implicit_als_training_batch Batch
 * @ingroup implicit_als_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of implicit ALS model-based training
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for implicit ALS model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of implicit ALS model-based training in the batch processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__BATCH"></a>
 * \brief Algorithm class for training the implicit ALS model
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for implicit ALS model training, double or float
 * \tparam method           Implicit ALS training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                Implicit ALS training method
 *      - \ref NumericTableInputId   Identifiers of input numeric table objects for the implicit ALS training algorithm
 *      - \ref ResultId              Identifiers of the results of the implicit ALS training algorithm
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Training<batch>
{
public:
    typedef algorithms::implicit_als::training::Input     InputType;
    typedef algorithms::implicit_als::Parameter           ParameterType;
    typedef algorithms::implicit_als::training::Result    ResultType;

    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< %Algorithm \ref implicit_als::interface1::Parameter "parameter" */

    /** Default constructor */
    Batch()
    {
        initialize();
    }

    /**
     * Constructs an implicit ALS training algorithm by copying input objects and parameters
     * of another implicit ALS training algorithm
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
     * Returns the structure that contains the results of the implicit ALS training algorithm
     * \return Structure that contains the results of the implicit ALS training algorithm
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store the results of the implicit ALS training algorithm
     * \param[in] res  Structure to store the results of the implicit ALS training algorithm
     */
    services::Status setResult(const ResultPtr& res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated implicit ALS training algorithm with a copy of input objects
     * of this implicit ALS training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

protected:
    training::ResultPtr _result;

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
        _result = training::ResultPtr(new ResultType());
    }
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

}
}
}
}

#endif
