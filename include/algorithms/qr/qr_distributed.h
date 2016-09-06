/* file: qr_distributed.h */
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
//  Implementation of the interface for the QR decomposition algorithm in the
//  distributed processing mode
//--
*/

#ifndef __QR_DISTRIBUTED_H__
#define __QR_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/qr/qr_types.h"
#include "algorithms/qr/qr_online.h"

namespace daal
{
namespace algorithms
{
namespace qr
{

namespace interface1
{
/** @defgroup qr_distributed Distributed
* @ingroup qr_without_pivoting
* @{
*/
/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the QR decomposition algorithm.
 *
 * \tparam step            Step of the QR decomposition algorithm in the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations of the QR decomposition algorithm, double or float
 * \tparam method           Computation method of the algorithm, \ref daal::algorithms::qr::Method
 *
 */
template<ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of QR decomposition algorithm on the first step in the distributed processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the QR decomposition algorithm, double or float
 * \tparam method           Computation method, \ref daal::algorithms::qr::Method
 *
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step1Local, algorithmFPType, method, cpu> : public
    OnlineContainer<algorithmFPType, method, cpu>
{
public:
    /** Default constructor */
    DistributedContainer(daal::services::Environment::env *daalEnv) : OnlineContainer<algorithmFPType, method, cpu>(daalEnv) {}
    /** Default destructor */
    virtual ~DistributedContainer() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the QR decomposition algorithm on the second step in the distributed processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the QR decomposition algorithm, double or float
 * \tparam method           Computation method, \ref daal::algorithms::qr::Method
 *
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step2Master, algorithmFPType, method, cpu> :
    public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the QR decomposition algorithm with a specified environment
     * in the second step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the QR decomposition algorithm in the second step of
     * the distributed processing mode
     */
    virtual void compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the QR decomposition algorithm in the second step of
     * the distributed processing mode
     */
    virtual void finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the QR decomposition algorithm on the third step in the distributed processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the QR decomposition algorithm, double or float
 * \tparam method           Computation method, \ref daal::algorithms::qr::Method
 *
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step3Local, algorithmFPType, method, cpu> :
    public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the QR decomposition algorithm with a specified environment
     * in the third step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the QR decomposition algorithm in the third step of
     * the distributed processing mode
     */
    virtual void compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the QR decomposition algorithm in the third step of
     * the distributed processing mode
     */
    virtual void finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTED"></a>
 * \brief Computes the results of the QR decomposition algorithm in the distributed processing mode.
 * \n<a href="DAAL-REF-QR-ALGORITHM">QR decomposition algorithm description and usage models</a>
 *
 * \tparam step             Step of the QR decomposition algorithm in the distributed processing mode
 * \tparam algorithmFPType  Data type to use in intermediate computations of the QR decomposition algorithm, double or float
 * \tparam method           Computation method, \ref daal::algorithms::qr::Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the QR decomposition algorithm
 */
template<ComputeStep step, typename algorithmFPType = double, Method method = defaultDense>
class DAAL_EXPORT Distributed : public daal::algorithms::Analysis<distributed> {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the result of the first step of the QR decomposition algorithm in the distributed processing mode.
 * \n<a href="DAAL-REF-QR-ALGORITHM">QR decomposition algorithm description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the QR decomposition algorithm, double or float
 * \tparam method           Computation method, \ref daal::algorithms::qr::Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step1Local, algorithmFPType, method> : public Online<algorithmFPType, method>
{
public:
    Distributed() : Online<algorithmFPType, method>() {}

    /**
     * Constructs a QR decomposition algorithm by copying input objects and parameters
     * of another QR decomposition algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step1Local, algorithmFPType, method> &other) :
        Online<algorithmFPType, method>(other)
    {}

    /**
     * Returns a pointer to the newly allocated QR decomposition algorithm
     * with a copy of input objects and parameters of this QR decomposition algorithm
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
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTED_STEP2MASTER_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the QR decomposition algorithm on the second step in the distributed processing mode.
 * \n<a href="DAAL-REF-QR-ALGORITHM">QR decomposition algorithm description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the QR decomposition algorithm, double or float
 * \tparam method           Computation method, \ref daal::algorithms::qr::Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step2Master, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef DistributedStep2Input Input;

    Input     input;     /*!< Input data structure */
    Parameter parameter; /*!< QR parameters structure */

    Distributed()
    {
        initialize();
    }

    /**
     * Constructs a QR decomposition algorithm by copying input objects and parameters
     * of another QR decomposition algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step2Master, algorithmFPType, method> &other)
    {
        initialize();
        input.set(inputOfStep2FromStep1, other.input.get(inputOfStep2FromStep1));
        parameter = other.parameter;
    }

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns structure that contains the results of the QR decomposition algorithm
     * \return Structure that contains the results of the QR decomposition algorithm
     */
    services::SharedPtr<Result> getResult()
    {
        return _partialResult->get(finalResultFromStep2Master);
    }

    /**
     * Returns structure that contains partial results of the QR decomposition algorithm
     * \return Structure that contains partial results of the QR decomposition algorithm
     */
    services::SharedPtr<DistributedPartialResult> getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Registers user-allocated memory to store the results of the QR decomposition algorithm
     * \return Structure to store the results of the QR decomposition algorithm
     */
    void setPartialResult(const services::SharedPtr<DistributedPartialResult>& partialRes)
    {
        DAAL_CHECK(partialRes->get(finalResultFromStep2Master), ErrorNullResult)
        _partialResult = partialRes;
        _pres = _partialResult.get();
    }

    /**
     * Checks parameters of the finalizeCompute() method
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
    }

    /**
     * Returns a pointer to the newly allocated QR decomposition algorithm
     * with a copy of input objects and parameters of this QR decomposition algorithm
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

    virtual void allocateResult() DAAL_C11_OVERRIDE {}

    virtual void allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult = services::SharedPtr<DistributedPartialResult>(new DistributedPartialResult());
        _partialResult->allocate<algorithmFPType>(_in, 0, 0);
        _pres = _partialResult.get();
    }

    virtual void initializePartialResult() DAAL_C11_OVERRIDE
    {
        _pres = _partialResult.get();
    }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, algorithmFPType, method)(&_env);
        _in   = &input;
        _par  = &parameter;
    }

private:
    services::SharedPtr<DistributedPartialResult> _partialResult;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTED_STEP3LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the QR decomposition algorithm on the third step in the distributed processing mode.
 * \n<a href="DAAL-REF-QR-ALGORITHM">QR decomposition algorithm description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the QR decomposition algorithm, double or float
 * \tparam method           Computation method, \ref daal::algorithms::qr::Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step3Local, algorithmFPType, method> : public
    daal::algorithms::Analysis<distributed>
{
public:
    typedef DistributedStep3Input Input;

    Input     input;     /*!< Input object */
    Parameter parameter; /*!< QR parameters */

    /** Default constructor */
    Distributed()
    {
        initialize();
    }

    /**
     * Constructs a QR decomposition algorithm by copying input objects and parameters
     * of another QR decomposition algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step3Local, algorithmFPType, method> &other)
    {
        initialize();
        input.set(inputOfStep3FromStep1, other.input.get(inputOfStep3FromStep1));
        input.set(inputOfStep3FromStep2, other.input.get(inputOfStep3FromStep2));
        parameter = other.parameter;
    }

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns the structure that contains the results of the QR decomposition algorithm
     * \return Structure that contains the results of the QR decomposition algorithm
     */
    services::SharedPtr<Result> getResult()
    {
        return _partialResult->get(finalResultFromStep3);
    }

    /**
     * Returns the structure that contains the results of the QR decomposition algorithm
     * \return Structure that contains the results of the QR decomposition algorithm
     */
    services::SharedPtr<DistributedPartialResultStep3> getPartialResult()
    {
        return _partialResult;
    }

    /**
    * Registers user-allocated memory to store the results of the QR decomposition algorithm
    * \return Structure to store the results of the QR decomposition algorithm
    */
    void setPartialResult(const services::SharedPtr<DistributedPartialResultStep3>& partialRes)
    {
        DAAL_CHECK(partialRes->get(finalResultFromStep3), ErrorNullResult)
        _partialResult = partialRes;
        _pres = _partialResult.get();
    }

    /**
    * Sets structure to store the results of the QR decomposition algorithm
    * \return Structure to store the results of the QR decomposition algorithm
    */
    void setResult(const services::SharedPtr<Result>& res) {}

    /**
     * Validates parameters of the finalizeCompute() method
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
    }

    /**
     * Returns a pointer to the newly allocated QR decomposition algorithm
     * with a copy of input objects and parameters of this QR decomposition algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step3Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step3Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step3Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step3Local, algorithmFPType, method>(*this);
    }

    virtual void allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult = services::SharedPtr<DistributedPartialResultStep3>(new DistributedPartialResultStep3());

        _partialResult->allocate<algorithmFPType>(_in, 0, 0);

        data_management::DataCollectionPtr qCollection = input.get(inputOfStep3FromStep1);

        _partialResult->setPartialResultStorage<algorithmFPType>(qCollection.get());

        _pres = _partialResult.get();
    }

    virtual void allocateResult() DAAL_C11_OVERRIDE {}

    virtual void initializePartialResult() DAAL_C11_OVERRIDE {}

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step3Local, algorithmFPType, method)(&_env);
        _in   = &input;
        _par  = &parameter;
    }

private:
    services::SharedPtr<DistributedPartialResultStep3> _partialResult;
};
/** @} */
} // namespace interface1
using interface1::DistributedContainer;
using interface1::Distributed;

} // namespace daal::algorithms::qr
} // namespace daal::algorithms
} // namespace daal
#endif
