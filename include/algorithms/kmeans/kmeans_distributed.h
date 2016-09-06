/* file: kmeans_distributed.h */
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
/** @defgroup kmeans_distributed Distributed
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
    virtual void compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the K-Means algorithm in the first step of the
     * distributed processing mode
     */
    virtual void finalizeCompute() DAAL_C11_OVERRIDE;
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
    virtual void compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the K-Means algorithm in the second step of the
     * distributed processing mode
     */
    virtual void finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTED"></a>
 * \brief Computes the results of the K-Means algorithm in the distributed processing mode
 * \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm description and usage models</a>
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
template<ComputeStep step, typename algorithmFPType = double, Method method = lloydDense>
class DAAL_EXPORT Distributed {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the K-Means algorithm in the first step of the distributed processing mode
 * \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of K-Means, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the K-Means algorithm
 *      - \ref InputId  Identifiers of input objects for the  K-Means algorithm
 *      - \ref ResultId Identifiers of results of the K-Means algorithm
 *
 * \par References
 *      - Input  class
 *      - Result class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step1Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
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
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store the results of the K-Means algorithm
     * \param[in] result  Structure to store the results of the K-Means algorithm
     */
    void setResult(const services::SharedPtr<Result>& result)
    {
        DAAL_CHECK(result, ErrorNullResult)
        _result = result;
        _res = _result.get();
    }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    services::SharedPtr<PartialResult> getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Sets the structure that contains computed partial results
     */
    void setPartialResult(const services::SharedPtr<PartialResult>& partialRes)
    {
        _partialResult = partialRes;
        _pres = _partialResult.get();
    }

    /**
     * Validates the parameters of the finalizeCompute() method
     */
    void checkFinalizeComputeParams() DAAL_C11_OVERRIDE {}

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

    virtual void allocateResult() DAAL_C11_OVERRIDE
    {
        _result = services::SharedPtr<Result>(new Result());
        _result->allocate<algorithmFPType>(_in, _par, (int) method);
        _res = _result.get();
    }

    virtual void allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult = services::SharedPtr<PartialResult>(new PartialResult());
        _partialResult->allocate<algorithmFPType>(&input, _par, (int) method);
        _pres = _partialResult.get();
    }

    virtual void initializePartialResult() DAAL_C11_OVERRIDE {}

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step1Local, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
    }

public:
    Input input;            /*!< %Input data structure */
    Parameter parameter;    /*!< K-Means parameters structure */

private:
    services::SharedPtr<PartialResult> _partialResult;
    services::SharedPtr<Result> _result;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTED_STEP2MASTER_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the K-Means algorithm in the second step of the distributed processing mode
 * \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm description and usage models</a>
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
 *      - Result class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step2Master, algorithmFPType, method>: public daal::algorithms::Analysis<distributed>
{
public:
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
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store the results of the K-Means algorithm
     * \param[in] result  Structure to store the results of the K-Means algorithm
     */
    void setResult(const services::SharedPtr<Result>& result)
    {
        DAAL_CHECK(result, ErrorNullResult)
        _result = result;
        _res = _result.get();
    }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    services::SharedPtr<PartialResult> getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Sets the structure that contains computed partial results
     */
    void setPartialResult(const services::SharedPtr<PartialResult>& partialRes)
    {
        _partialResult = partialRes;
        _pres = _partialResult.get();
    }

    /**
     * Validates the parameters of the finalizeCompute() method
     */
    void checkFinalizeComputeParams() DAAL_C11_OVERRIDE
    {
        if(_partialResult)
        {
            _partialResult->check(_par, method);
            if (!_errors->isEmpty()) { return; }
        }
        else
        {
            _errors->add(services::ErrorNullResult);
            return;
        }

        if(_result)
        {
            _result->check(_partialResult.get(), _par, method);
        }
        else
        {
            _errors->add(services::ErrorNullResult);
            return;
        }
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

    virtual void allocateResult() DAAL_C11_OVERRIDE
    {
        _result = services::SharedPtr<Result>(new Result());
        _result->allocate<algorithmFPType>(_pres, _par, (int) method);
        _res = _result.get();
    }

    virtual void allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult = services::SharedPtr<PartialResult>(new PartialResult());
        _partialResult->allocate<algorithmFPType>(&input, _par, (int) method);
        _pres = _partialResult.get();
    }

    virtual void initializePartialResult() DAAL_C11_OVERRIDE {}

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
    }

public:
    DistributedStep2MasterInput input; /*!< %Input data structure */
    Parameter parameter;               /*!< K-Means parameters structure */

private:
    services::SharedPtr<PartialResult> _partialResult;
    services::SharedPtr<Result> _result;
};
} // namespace interface1
using interface1::DistributedContainer;
using interface1::Distributed;
/** @} */
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
#endif
