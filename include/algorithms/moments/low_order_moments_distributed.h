/* file: low_order_moments_distributed.h */
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
//  Implementation of the interface for the low order moments algorithm in the
//  distributed processing mode
//--
*/

#ifndef __LOW_ORDER_MOMENTS_DISTRIBUTED_H__
#define __LOW_ORDER_MOMENTS_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
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
/** @defgroup low_order_moments_distributed Distributed
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
 * \brief Provides methods to run implementations of the first step of the low order moments algorithm
 *        in the distributed processing mode.
 *        This class is associated with daal::algorithms::low_order_moments::Distributed class.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the low order moments, double or float
 * \tparam method           Computation method, \ref daal::algorithms::low_order_moments::Method
 *
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step1Local, algorithmFPType, method, cpu> :
    public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the low order moments algorithm with a specified environment
     * in the first step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the low order moments algorithm
     * in the first step of the distributed processing mode
     */
    virtual void compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the low order moments algorithm
     * in the first step of the distributed processing mode
     */
    virtual void finalizeCompute() DAAL_C11_OVERRIDE;
};

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
    virtual void compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the low order moments algorithm
     * in the second step of the distributed processing mode
     */
    virtual void finalizeCompute() DAAL_C11_OVERRIDE;
};


/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTED"></a>
 * \brief Computes moments of low order in the distributed processing mode.
 * \n<a href="DAAL-REF-LOW_ORDER_MOMENTS-ALGORITHM">Low order moments algorithm description and usage models</a>
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
template<ComputeStep step, typename algorithmFPType = double, Method method = defaultDense>
class DAAL_EXPORT Distributed : public daal::algorithms::Analysis<distributed> {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the result of the first step of the moments of low order algorithm
 *        in the distributed processing mode.
 * \n<a href="DAAL-REF-LOW_ORDER_MOMENTS-ALGORITHM">Low order moments algorithm description and usage models</a>
 *
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
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step1Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    DistributedInput<step1Local> input;  /*!< Input data structure */

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
    Distributed(const Distributed<step1Local, algorithmFPType, method> &other)
    {
        initialize();
        input.set(data,  other.input.get(data));
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
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store final results of the low order moments algorithm
     * \param[in] result    Structure for storing the results of the low order moments algorithm
     */
    void setResult(const services::SharedPtr<Result> &result)
    {
        DAAL_CHECK(result, ErrorNullResult)
        _result = result;
        _res = _result.get();
    }

    /**
     * Returns the structure that contains partial results of the low order moments algorithm
     * \return Structure that contains partial results
     */
    services::SharedPtr<PartialResult> getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Registers user-allocated memory to store partial results of the low order moments algorithm
     * \param[in] partialResult    Structure for storing partial results of the low order moments algorithm
     * \param[in] initFlag         Flag that specifies whether the partial results are initialized
     */
    void setPartialResult(const services::SharedPtr<PartialResult> &partialResult, bool initFlag = false)
    {
        _partialResult = partialResult;
        _pres = _partialResult.get();
        setInitFlag(initFlag);
    }

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

    virtual void allocateResult() DAAL_C11_OVERRIDE
    {
        _result->allocate<algorithmFPType>(_pres, 0, 0);
        _res    = _result.get();
    }

    virtual void allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult->allocate<algorithmFPType>(_in, 0, 0);
        _pres   = _partialResult.get();
    }

    virtual void initializePartialResult() DAAL_C11_OVERRIDE
    {}

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step1Local, algorithmFPType, method)(&_env);
        _in = &input;
        _par = NULL;
        _result = services::SharedPtr<Result>(new Result());
        _partialResult = services::SharedPtr<PartialResult>(new PartialResult());
    }

private:
    services::SharedPtr<PartialResult> _partialResult;
    services::SharedPtr<Result> _result;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTED_STEP2MASTER_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the result of the second step of the moments of low order algorithm
 *        in the distributed processing mode.
 * \n<a href="DAAL-REF-LOW_ORDER_MOMENTS-ALGORITHM">Low order moments algorithm description and usage models</a>
 *
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
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step2Master, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    DistributedInput<step2Master> input;  /*!< Input data structure */

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
    Distributed(const Distributed<step2Master, algorithmFPType, method> &other)
    {
        initialize();
        input.set(partialResults, other.input.get(partialResults));
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
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store final results of the low order moments algorithm
     * \param[in] result    Structure for storing the results of the low order moments algorithm
     */
    void setResult(const services::SharedPtr<Result> &result)
    {
        DAAL_CHECK(result, ErrorNullResult)
        _result = result;
        _res = _result.get();
    }

    /**
     * Returns the structure that contains partial results of the low order moments algorithm
     * \return Structure that contains partial results
     */
    services::SharedPtr<PartialResult> getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Registers user-allocated memory to store partial results of the low order moments algorithm
     * \param[in] partialResult    Structure for storing partial results of the low order moments algorithm
     * \param[in] initFlag         Flag that specifies whether the partial results are initialized
     */
    void setPartialResult(const services::SharedPtr<PartialResult> &partialResult, bool initFlag = false)
    {
        _partialResult = partialResult;
        _pres = _partialResult.get();
        setInitFlag(initFlag);
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

    virtual void allocateResult() DAAL_C11_OVERRIDE
    {
        _result->allocate<algorithmFPType>(_pres, 0, 0);
        _res    = _result.get();
    }

    virtual void allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult->allocate<algorithmFPType>(_in, 0, 0);
        _pres   = _partialResult.get();
    }

    virtual void initializePartialResult() DAAL_C11_OVERRIDE
    {}

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, algorithmFPType, method)(&_env);
        _in = &input;
        _par = NULL;
        _result = services::SharedPtr<Result>(new Result());
        _partialResult = services::SharedPtr<PartialResult>(new PartialResult());
    }

private:
    services::SharedPtr<PartialResult> _partialResult;
    services::SharedPtr<Result> _result;
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
