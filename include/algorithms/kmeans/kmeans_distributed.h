/* file: kmeans_distributed.h */
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
//  Implementation of the interface for the K-Means algorithm in the distributed
//  processing mode
//--
*/

#ifndef __KMEANS_DISTRIBITED_H__
#define __KMEANS_DISTRIBITED_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/kmeans/kmeans_types.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{

namespace interface1
{
/**
 * @defgroup kmeans_distributed Distributed
 * @ingroup kmeans_compute
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the K-Means algorithm.
 *        This class is associated with the daal::algorithms::kmeans::Distributed class
 *        and supports the method of K-Means computation in the distributed processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of K-Means, double or float
 * \tparam method           Computation method of the algorithm, \ref daal::algorithms::kmeans::Method
 */
template<ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the K-Means algorithm in the first step of the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step1Local, algorithmFPType, method, cpu> : public
    daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the K-Means algorithm with a specified environment
     * in the first step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the K-Means algorithm in the first step of the
     * distributed processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the K-Means algorithm in the first step of the
     * distributed processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTEDCONTAINER_STEP2MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the K-Means algorithm in the second step of the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step2Master, algorithmFPType, method, cpu> : public
    daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the K-Means algorithm with a specified environment
     * in the second step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the K-Means algorithm in the second step of the
     * distributed processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the K-Means algorithm in the second step of the
     * distributed processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTED"></a>
 * \brief Computes the results of the K-Means algorithm in the distributed processing mode
 * <!-- \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of K-Means, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the K-Means algorithm
 *      - \ref InputId  Identifiers of input objects for the K-Means algorithm
 *      - \ref ResultId Identifiers of results of the K-Means algorithm
 *
 * \par References
 *      - Input class
 *      - Result class
 */
template<ComputeStep step, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = lloydDense>
class DAAL_EXPORT Distributed {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the K-Means algorithm in the first step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of K-Means, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the K-Means algorithm
 *      - \ref InputId  Identifiers of input objects for the  K-Means algorithm
 *      - \ref ResultId Identifiers of results of the K-Means algorithm
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step1Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::kmeans::Input         InputType;
    typedef algorithms::kmeans::Parameter     ParameterType;
    typedef algorithms::kmeans::Result        ResultType;
    typedef algorithms::kmeans::PartialResult PartialResultType;

    /**
     * Constructs a K-Means algorithm
     *  \param[in] nClusters  Number of clusters
     *  \param[in] assignFlag Flag to calculate partial assignment
     */
    Distributed(size_t nClusters, bool assignFlag = false) : parameter(nClusters, 1)
    {
        initialize();
        parameter.assignFlag = assignFlag;
    }

    /**
     * Constructs a K-Means algorithm by copying input objects and parameters
     * of another K-Means algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step1Local, algorithmFPType, method> &other) : parameter(other.parameter)
    {
        initialize();
        input.set(data,           other.input.get(data));
        input.set(inputCentroids, other.input.get(inputCentroids));
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains the results of the K-Means algorithm
     * \return Structure that contains the results of the K-Means algorithm
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store the results of the K-Means algorithm
     * \param[in] result  Structure to store the results of the K-Means algorithm
     */
    services::Status setResult(const ResultPtr& result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    PartialResultPtr getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const PartialResultPtr& partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres = _partialResult.get();
        return services::Status();
    }

    /**
     * Validates the parameters of the finalizeCompute() method
     */
    services::Status checkFinalizeComputeParams() DAAL_C11_OVERRIDE
    {
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated K-Means algorithm with a copy of input objects
     * and parameters of this K-Means algorithm
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

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        services::Status s = _result->allocate<algorithmFPType>(_in, _par, (int) method);
        _res = _result.get();
        return s;
    }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult.reset(new PartialResultType());
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int) method);
        _pres = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        return services::Status();
    }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step1Local, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
    }

public:
    InputType input;            /*!< %Input data structure */
    ParameterType parameter;    /*!< K-Means parameters structure */

private:
    PartialResultPtr _partialResult;
    ResultPtr _result;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTED_STEP2MASTER_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the K-Means algorithm in the second step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of K-Means, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the K-Means algorithm
 *      - \ref InputId  Identifiers of input objects for the K-Means algorithm
 *      - \ref ResultId Identifiers of results of the K-Means algorithm
 *
 * \par References
 *      - Input  class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step2Master, algorithmFPType, method>: public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::kmeans::DistributedStep2MasterInput InputType;
    typedef algorithms::kmeans::Parameter                   ParameterType;
    typedef algorithms::kmeans::Result                      ResultType;
    typedef algorithms::kmeans::PartialResult               PartialResultType;

    /**
     *  Main constructor
     *  \param[in] nClusters   Number of clusters
     *  \param[in] nIterations Number of iterations
     */
    Distributed(size_t nClusters, size_t nIterations = 1) : parameter(nClusters, nIterations)
    {
        initialize();
        parameter.assignFlag = false;
    }

    /**
     * Constructs a K-Means algorithm by copying input objects and parameters
     * of another K-Means algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step2Master, algorithmFPType, method> &other) : parameter(other.parameter)
    {
        initialize();
        input.set(partialResults, other.input.get(partialResults));
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains the results of the K-Means algorithm
     * \return Structure that contains the results of the K-Means algorithm
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store the results of the K-Means algorithm
     * \param[in] result  Structure to store the results of the K-Means algorithm
     */
    services::Status setResult(const ResultPtr& result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    PartialResultPtr getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const PartialResultPtr& partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres = _partialResult.get();
        return services::Status();
    }

    /**
     * Validates the parameters of the finalizeCompute() method
     */
    services::Status checkFinalizeComputeParams() DAAL_C11_OVERRIDE
    {
        services::Status s;
        if(_partialResult)
        {
            s |= _partialResult->check(_par, method);
            if (!s) { return s; }
        }
        else
        {
            return services::Status(services::ErrorNullResult);
        }

        if(_result)
        {
            s |= _result->check(_partialResult.get(), _par, method);
        }
        else
        {
            return services::Status(services::ErrorNullResult);
        }
        return s;
    }

    /**
     * Returns a pointer to the newly allocated K-Means algorithm with a copy of input objects
     * and parameters of this K-Means algorithm
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
        _result.reset(new ResultType());
        services::Status s = _result->allocate<algorithmFPType>(_pres, _par, (int) method);
        _res = _result.get();
        return s;
    }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult.reset(new PartialResultType());
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int) method);
        _pres = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        return services::Status();
    }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
    }

public:
    InputType input;                /*!< %Input data structure */
    ParameterType parameter;        /*!< K-Means parameters structure */

private:
    PartialResultPtr _partialResult;
    ResultPtr _result;
};
/** @} */
} // namespace interface1
using interface1::DistributedContainer;
using interface1::Distributed;
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
#endif
