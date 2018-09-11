/* file: low_order_moments_distributed.h */
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
//  Implementation of the interface for the low order moments algorithm in the
//  distributed processing mode
//--
*/

#ifndef __LOW_ORDER_MOMENTS_DISTRIBUTED_H__
#define __LOW_ORDER_MOMENTS_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "services/daal_defines.h"
#include "algorithms/moments/low_order_moments_types.h"

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{

namespace interface1
{
/**
 * @defgroup low_order_moments_distributed Distributed
 * @ingroup low_order_moments
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTEDCONTAINER_STEP_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Provides methods to run implementations of the low order moments algorithm in the distributed processing mode.
 *        This class is associated with daal::algorithms::low_order_moments::Distributed class
 *
 * \tparam step             Step of distributed processing, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations of the low order moments, double or float
 * \tparam method           Computation method, \ref daal::algorithms::low_order_moments::Method
 *
 */
template<ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer
{};

/**
 * \brief Provides methods to run implementations of the second step of the low order moments algorithm
 *        in the distributed processing mode.
 *        This class is associated with daal::algorithms::low_order_moments::Distributed class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the low order moments, double or float
 * \tparam method           Computation method, \ref daal::algorithms::low_order_moments::Method
 *
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step2Master, algorithmFPType, method, cpu> :
    public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the low order moments algorithm with a specified environment
     * in the second step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the low order moments algorithm
     * in the second step of the distributed processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the low order moments algorithm
     * in the second step of the distributed processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};


/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTED"></a>
 * \brief Computes moments of low order in the distributed processing mode.
 * <!-- \n<a href="DAAL-REF-LOW_ORDER_MOMENTS-ALGORITHM">Low order moments algorithm description and usage models</a> -->
 *
 * \tparam step            Step of distributed processing, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations of the low order moments, double or float
 * \tparam method           Computation method, \ref daal::algorithms::low_order_moments::Method
 *
 * \par Enumerations
 *      - \ref Method           Computation methods for the low order moments algorithm
 *      - \ref InputId          Identifiers of input objects for the low order moments algorithm
 *      - \ref PartialResultId  Identifiers of partial results of the low order moments algorithm
 *      - \ref ResultId         Identifiers of the results of the low order moments algorithm *
 * \par References
 *      - Input class
 *      - PartialResult class
 *      - Result class
 */
template<ComputeStep step, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Distributed{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the result of the first step of the moments of low order algorithm
 *        in the distributed processing mode.
 * <!-- \n<a href="DAAL-REF-LOW_ORDER_MOMENTS-ALGORITHM">Low order moments algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the low order moments, double or float
 * \tparam method           Computation method, \ref daal::algorithms::low_order_moments::Method
 *
 * \par Enumerations
 *      - \ref Method           Computation methods for the low order moments algorithm
 *      - \ref InputId          Identifiers of input objects for the low order moments algorithm
 *      - \ref PartialResultId  Identifiers of partial results of the low order moments algorithm
 *      - \ref ResultId         Identifiers of the results of the low order moments algorithm
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step1Local, algorithmFPType, method> : public Online<algorithmFPType, method>
{
public:
    typedef Online<algorithmFPType, method> super;

    typedef typename super::InputType         InputType;
    typedef typename super::ParameterType     ParameterType;
    typedef typename super::ResultType        ResultType;
    typedef typename super::PartialResultType PartialResultType;

    /** Default constructor */
    Distributed()
    {}

    /**
     * Constructs an algorithm that computes moments of low order by copying input objects
     * of another algorithm that computes moments of low order
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step1Local, algorithmFPType, method> &other) : Online<algorithmFPType, method>(other)
    {}

    /**
     * Returns a pointer to the newly allocated algorithm that computes moments of low order
     * with a copy of input objects of this algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step1Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step1Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step1Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step1Local, algorithmFPType, method>(*this);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTED_STEP2MASTER_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the result of the second step of the moments of low order algorithm
 *        in the distributed processing mode.
 * <!-- \n<a href="DAAL-REF-LOW_ORDER_MOMENTS-ALGORITHM">Low order moments algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the low order moments, double or float
 * \tparam method           Computation method, \ref daal::algorithms::low_order_moments::Method
 *
 * \par Enumerations
 *      - \ref Method           Computation methods for the low order moments algorithm
 *      - \ref InputId          Identifiers of input objects for the low order moments algorithm
 *      - \ref PartialResultId  Identifiers of partial results of the low order moments algorithm
 *      - \ref ResultId         Identifiers of the results of the low order moments algorithm
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step2Master, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::low_order_moments::DistributedInput<step2Master> InputType;
    typedef algorithms::low_order_moments::Parameter                     ParameterType;
    typedef algorithms::low_order_moments::Result                        ResultType;
    typedef algorithms::low_order_moments::PartialResult                 PartialResultType;

    DistributedInput<step2Master> input;                /*!< Input data structure */
    ParameterType parameter;        /*!< %Parameters structure */

    /** Default constructor */
    Distributed()
    {
        initialize();
    }

    /**
     * Constructs an algorithm that computes moments of low order by copying input objects
     * of another algorithm that computes moments of low order
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step2Master, algorithmFPType, method> &other) : parameter(other.parameter), input(other.input)
    {
        initialize();
    }

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns structure that contains final results of the low order moments algorithm
     * \return Structure that contains final results of the low order moments algorithm
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store final results of the low order moments algorithm
     * \param[in] result    Structure for storing the results of the low order moments algorithm
     */
    services::Status setResult(const ResultPtr &result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains partial results of the low order moments algorithm
     * \return Structure that contains partial results
     */
    PartialResultPtr getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Registers user-allocated memory to store partial results of the low order moments algorithm
     * \param[in] partialResult    Structure for storing partial results of the low order moments algorithm
     * \param[in] initFlag         Flag that specifies whether the partial results are initialized
     */
    services::Status setPartialResult(const PartialResultPtr &partialResult, bool initFlag = false)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult);
        _partialResult = partialResult;
        _pres = _partialResult.get();
        setInitFlag(initFlag);
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated algorithm that computes moments of low order
     * with a copy of input objects of this algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step2Master, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step2Master, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step2Master, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step2Master, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(_pres, 0, 0);
        _res    = _result.get();
        return s;
    }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(_in, 0, 0);
        _pres   = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        return services::Status();
    }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _result.reset(new ResultType());
        _partialResult.reset(new PartialResultType());
    }

private:
    PartialResultPtr _partialResult;
    ResultPtr _result;
};
/** @} */
} // namespace interface1
using interface1::DistributedInput;
using interface1::DistributedContainer;
using interface1::Distributed;

} // namespace daal::algorithms::low_order_moments
}
}
#endif
