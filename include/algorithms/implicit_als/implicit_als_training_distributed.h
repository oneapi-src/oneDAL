/* file: implicit_als_training_distributed.h */
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
//  Implementation of the interface for implicit ALS model-based training in the
//  distributed processing mode
//--
*/

#ifndef __IMPLICIT_ALS_TRAINING_DISTRIBUTED_H__
#define __IMPLICIT_ALS_TRAINING_DISTRIBUTED_H__

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
 * @defgroup implicit_als_training_distributed Distributed
 * @ingroup implicit_als_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDCONTAINER"></a>
 * \brief Class containing methods to compute the result of implicit ALS model-based training
 * in the distributed processing mode
 */
template<ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods to train the implicit ALS model in the first step of the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step1Local, algorithmFPType, method, cpu> : public
    TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for implicit ALS model-based training with a specified environment
     * in the first step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
     /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of implicit ALS model-based training
     * in the first step of the distributed processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of implicit ALS model-based training
     * in the first step of the distributed processing mode
     */
    void finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDCONTAINER_STEP2MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods to train the implicit ALS model in the second step of the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step2Master, algorithmFPType, method, cpu> : public
    TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for implicit ALS model-based training with a specified environment
     * in the second step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
     /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of implicit ALS model-based training
     * in the second step of the distributed processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of implicit ALS model-based training
     * in the second step of the distributed processing mode
     */
    void finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDCONTAINER_STEP3LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods to train the implicit ALS model in the third step of the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step3Local, algorithmFPType, method, cpu> : public
    TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for implicit ALS model-based training with a specified environment
     * in the third step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
     /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of implicit ALS model-based training
     * in the third step of the distributed processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of implicit ALS model-based training
     * in the third step of the distributed processing mode
     */
    void finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDCONTAINER_STEP4LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods to train the implicit ALS model in the fourth step of the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step4Local, algorithmFPType, method, cpu> : public
    TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for implicit ALS model-based training with a specified environment
     * in the fourth step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
     /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of implicit ALS model-based training
     * in the fourth step of the distributed processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of implicit ALS model-based training
     * in the fourth step of the distributed processing mode
     */
    void finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTED"></a>
 * \brief Trains the implicit ALS model in the distributed processing mode
 *
 * \tparam step             Step of the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for the implicit ALS training algorithm
 *                          in the distributed processing mode, double or float
 * \tparam method           Implicit ALS training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method     %Training methods of the implicit ALS algorithm in the first step of the distributed processing mode
 */
template<ComputeStep step, typename algorithmFPType = double, Method method = fastCSR>
class DAAL_EXPORT Distributed : public Training<distributed> {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Trains the implicit ALS model in the first step of the distributed processing mode
 *
 * \tparam step             Step of the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for the implicit ALS training algorithm in the first step
 *                          of the distributed processing mode, double or float
 * \tparam method           Implicit ALS training method, \ref Method
 *
 * \par Enumerations
 *      - Method  %Training methods of the implicit ALS algorithm in the first step of the distributed processing mode
 *
 * \par References
 *      - \ref implicit_als::interface1::Parameter "implicit_als::Parameter" class
 *      - \ref DistributedInput<step1Local> class
 *      - \ref DistributedPartialResultStep1 class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step1Local, algorithmFPType, method> : public Training<distributed>
{
public:
    DistributedInput<step1Local> input;  /*!< %Input data structure */
    Parameter parameter; /*!< Training parameters */

    /** Default constructor */
    Distributed()
    {
        initialize();
    }

    /**
     * Constructs an implicit ALS training algorithm by copying input objects and parameters
     * of another implicit ALS training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step1Local, algorithmFPType, method> &other)
    {
        initialize();
        input.set(partialModel, other.input.get(partialModel));
        parameter = other.parameter;
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store partial results of the implicit ALS training algorithm
     * in the first step of the distributed processing mode
     * \param[in] partialResult  Structure to store partial results of the implicit ALS training algorithm
     * in the first step of the distributed processing mode
     */
    void setPartialResult(const services::SharedPtr<DistributedPartialResultStep1>& partialResult)
    {
        _partialResult = partialResult;
        _pres = _partialResult.get();
    }

    /**
     * Returns the structure that contains partial results of the implicit ALS training algorithm
     * in the first step of the distributed processing mode
     * \return Structure that contains partial results of the implicit ALS training algorithm
     * in the first step of the distributed processing mode
     */
    services::SharedPtr<DistributedPartialResultStep1> getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to the newly allocated ALS training algorithm with a copy of input objects
     * and parameters of this ALS training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step1Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step1Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    services::SharedPtr<DistributedPartialResultStep1> _partialResult;

    virtual Distributed<step1Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step1Local, algorithmFPType, method>(*this);
    }

    void allocateResult() DAAL_C11_OVERRIDE
    {}

    void allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult->allocate<algorithmFPType>(&input, &parameter, method);
        _pres = _partialResult.get();
    }

    void initializePartialResult() DAAL_C11_OVERRIDE
    {}

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step1Local, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _partialResult = services::SharedPtr<DistributedPartialResultStep1>(new DistributedPartialResultStep1());
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTED_STEP2MASTER_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Trains the implicit ALS model in the second step of the distributed processing mode
 *
 * \tparam step             Step of the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for the implicit ALS training algorithm in the second step
 *                          of the distributed processing mode, double or float
 * \tparam method           Implicit ALS training method, \ref Method
 *
 * \par Enumerations
 *      - Method  %Training methods of the implicit ALS algorithm in the second step of the distributed processing mode
 *
 * \par References
 *      - \ref implicit_als::interface1::Parameter "implicit_als::Parameter" class
 *      - \ref DistributedInput<step2Master> class
 *      - \ref DistributedPartialResultStep2 class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step2Master, algorithmFPType, method> : public Training<distributed>
{
public:
    DistributedInput<step2Master> input;  /*!< %Input data structure */
    Parameter parameter; /*!< Training parameters */

    /** Default constructor */
    Distributed()
    {
        initialize();
    }

    /**
     * Constructs an implicit ALS training algorithm by copying input objects and parameters
     * of another implicit ALS training algorithm
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
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store partial results of the implicit ALS training algorithm
     * in the second step of the distributed processing mode
     * \param[in] partialResult  Structure to store partial results of the implicit ALS training algorithm\
     * in the second step of the distributed processing mode
     */
    void setPartialResult(const services::SharedPtr<DistributedPartialResultStep2>& partialResult)
    {
        _partialResult = partialResult;
        _pres = _partialResult.get();
    }

    /**
     * Returns the structure that contains partial results of the implicit ALS training algorithm
     * in the second step of the distributed processing mode
     * \return Structure that contains partial results of the implicit ALS training algorithm
     * in the second step of the distributed processing mode
     */
    services::SharedPtr<DistributedPartialResultStep2> getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to the newly allocated ALS training algorithm with a copy of input objects
     * and parameters of this ALS training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step2Master, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step2Master, algorithmFPType, method> >(cloneImpl());
    }

protected:
    services::SharedPtr<DistributedPartialResultStep2> _partialResult;

    virtual Distributed<step2Master, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step2Master, algorithmFPType, method>(*this);
    }

    void allocateResult() DAAL_C11_OVERRIDE
    {}

    void allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult->allocate<algorithmFPType>(&input, &parameter, method);
        _pres = _partialResult.get();
    }

    void initializePartialResult() DAAL_C11_OVERRIDE
    {}

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _partialResult = services::SharedPtr<DistributedPartialResultStep2>(new DistributedPartialResultStep2());
    }
};


/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTED_STEP3LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Trains the implicit ALS model in the third step of the distributed processing mode
 *
 * \tparam step             Step of the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for the implicit ALS training algorithm in the third step
 *                          of the distributed processing mode, double or float
 * \tparam method           Implicit ALS training method, Method
 *
 * \par Enumerations
 *      - Method  %Training methods of the implicit ALS algorithm in the third step of the distributed processing mode
 *
 * \par References
 *      - \ref implicit_als::interface1::Parameter "implicit_als::Parameter" class
 *      - \ref DistributedInput<step3Local> class
 *      - \ref DistributedPartialResultStep3 class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step3Local, algorithmFPType, method> : public Training<distributed>
{
public:
    DistributedInput<step3Local> input;  /*!< %Input data structure */
    Parameter parameter; /*!< Training parameters */

    /** Default constructor */
    Distributed()
    {
        initialize();
    }

    /**
     * Constructs an implicit ALS training algorithm by copying input objects and parameters
     * of another implicit ALS training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step3Local, algorithmFPType, method> &other)
    {
        initialize();
        input.set(partialModel,             other.input.get(partialModel));
        input.set(partialModelBlocksToNode, other.input.get(partialModelBlocksToNode));
        input.set(offset,                   other.input.get(offset));
        parameter = other.parameter;
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store partial results of the implicit ALS training algorithm
     * in the third step of the distributed processing mode
     * \param[in] partialResult  Structure to store partial results of the implicit ALS training algorithm
     * in the third step of the distributed processing mode
     */
    void setPartialResult(const services::SharedPtr<DistributedPartialResultStep3>& partialResult)
    {
        _partialResult = partialResult;
        _pres = _partialResult.get();
    }

    /**
     * Returns the structure that contains partial results of the implicit ALS training algorithm
     * in the third step of the distributed processing mode
     * \return Structure that contains partial results of the implicit ALS training algorithm
     * in the third step of the distributed processing mode
     */
    services::SharedPtr<DistributedPartialResultStep3> getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to the newly allocated ALS training algorithm with a copy of input objects
     * and parameters of this ALS training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step3Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step3Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    services::SharedPtr<DistributedPartialResultStep3> _partialResult;

    virtual Distributed<step3Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step3Local, algorithmFPType, method>(*this);
    }

    void allocateResult() DAAL_C11_OVERRIDE
    {}

    void allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult->allocate<algorithmFPType>(&input, &parameter, method);
        _pres = _partialResult.get();
    }

    void initializePartialResult() DAAL_C11_OVERRIDE
    {}

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step3Local, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _partialResult = services::SharedPtr<DistributedPartialResultStep3>(new DistributedPartialResultStep3());
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTED_STEP4LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Trains the implicit ALS model in the fourth step of the distributed processing mode
 *
 * \tparam step             Step of the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for the implicit ALS training algorithm in the fourth step
 *                          of the distributed processing mode, double or float
 * \tparam method           Implicit ALS training method, \ref Method
 *
 * \par Enumerations
 *      - Method  %Training methods of the implicit ALS algorithm in the fourth step of the distributed processing mode
 *
 * \par References
 *      - \ref implicit_als::interface1::Parameter "implicit_als::Parameter" class
 *      - \ref DistributedInput<step4Local> class
 *      - \ref DistributedPartialResultStep4 class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step4Local, algorithmFPType, method> : public Training<distributed>
{
public:
    DistributedInput<step4Local> input;  /*!< %Input data structure */
    Parameter parameter; /*!< Training parameters */

    /** Default constructor */
    Distributed()
    {
        initialize();
    }

    /**
     * Constructs an implicit ALS training algorithm by copying input objects and parameters
     * of another implicit ALS training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step4Local, algorithmFPType, method> &other)
    {
        initialize();
        input.set(partialModels,         other.input.get(partialModels));
        input.set(partialData,           other.input.get(partialData));
        input.set(inputOfStep4FromStep2, other.input.get(inputOfStep4FromStep2));
        parameter = other.parameter;
    }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store partial results of the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     * \param[in] partialResult  Structure to store partial results of the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     */
    void setPartialResult(const services::SharedPtr<DistributedPartialResultStep4>& partialResult)
    {
        _partialResult = partialResult;
        _pres = _partialResult.get();
    }

    /**
     * Returns the structure that contains partial results of the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     * \return Structure that contains partial results of the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     */
    services::SharedPtr<DistributedPartialResultStep4> getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to the newly allocated ALS training algorithm with a copy of input objects
     * and parameters of this ALS training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step4Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step4Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    services::SharedPtr<DistributedPartialResultStep4> _partialResult;

    virtual Distributed<step4Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step4Local, algorithmFPType, method>(*this);
    }

    void allocateResult() DAAL_C11_OVERRIDE
    {}

    void allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult->allocate<algorithmFPType>(&input, &parameter, method);
        _pres = _partialResult.get();
    }

    void initializePartialResult() DAAL_C11_OVERRIDE
    {}

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step4Local, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _partialResult = services::SharedPtr<DistributedPartialResultStep4>(new DistributedPartialResultStep4());
    }
};
/** @} */
} // namespace interface1
using interface1::DistributedContainer;
using interface1::Distributed;

}
}
}
}

#endif
