/* file: gbt_regression_init_distributed.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of the interface for initializing gradient boosted trees
//  regression training algorithm in the distributed processing mode
//--
*/

#ifndef __GBT_REGRESSION_INIT_DISTRIBUTED_H__
#define __GBT_REGRESSION_INIT_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_init_types.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_model.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
/**
 * @defgroup gbt_regression_training Training
 * \copydoc daal::algorithms::gbt::regression::training
 * @ingroup gbt_regression
 * @{
 */
/**
 * \brief Contains classes for Gradient Boosted Trees models training
 */
namespace init
{
namespace interface1
{
/**
 * @defgroup gbt_regression_training_init_distributed Distributed
 * @ingroup gbt_regression_training_init
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of initialization of ... algorithm.
 *        This class is associated with the daal::algorithms::gbt::regression::trianing::init::Distributed class
 *        and supports the method of computing ... for ... algorithm in the distributed processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of initial clusters for ... algorithm, double or float
 * \tparam method           Method of computing initial clusters for the algorithm, \ref daal::algorithms::gbt::regression::trianing::init::Method
 */
template<ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step1Local, algorithmFPType, method, cpu> : public
    TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for model-based training with a specified environment
     * in the first step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of model-based training
     * in the first step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of model-based training
     * in the first step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__DISTRIBUTEDCONTAINER_STEP2MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step2Master, algorithmFPType, method, cpu> : public
    TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for model-based training with a specified environment
     * in the first step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of model-based training
     * in the first step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of model-based training
     * in the first step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT_REGRESSION_TRAINING_INIT__DISTRIBUTEDCONTAINER_STEP3LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step3Local, algorithmFPType, method, cpu> : public
    TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for model-based training with a specified environment
     * in the first step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of model-based training
     * in the first step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of model-based training
     * in the first step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__INIT__DISTRIBUTED"></a>
 * \brief Computes the results of ... algorithm in the distributed processing mode
 * <!-- \n<a href="DAAL-REF-...-ALGORITHM">... algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of ..., double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for ... algorithm
 *      - \ref InputId  Identifiers of input objects for ... algorithm
 *      - \ref ResultId Identifiers of results of ... algorithm
 *
 * \par References
 *      - Input class
 *      - Result class
 */
template<ComputeStep step, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Distributed {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__INIT__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Provides methods for model-based training in the first step of distributed processing mode
 * <!-- \n<a href="DAAL-REF-GBT__REGRESSION__TRAINING__INIT-ALGORITHM">gradient boosted trees algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for model-based training, double or float
 * \tparam method           gradient boosted trees training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref gbt::regression::interface1::Model "Model" class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step1Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::gbt::regression::init::DistributedInput<step1Local>  InputType;
    typedef algorithms::gbt::regression::init::Parameter                     ParameterType;
    typedef algorithms::gbt::regression::init::DistributedPartialResultStep1 PartialResultType;

    /**
     * Constructs a gradient boosted trees training algorithm
     */
    Distributed(size_t _maxBins = 256);

    /**
     * Constructs a gradient boosted trees training algorithm by copying input objects
     * and parameters of another gradient boosted trees training algorithm in the first step of distributed processing mode
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step1Local, algorithmFPType, method> &other);

    ~Distributed()
    {
        delete _par;
    }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType& parameter() { return *static_cast<ParameterType*>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType& parameter() const { return *static_cast<const ParameterType*>(_par); }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    DistributedPartialResultStep1Ptr getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep1Ptr& partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated gradient boosted trees training algorithm with a copy of input objects
     * and parameters of this gradient boosted trees training algorithm algorithm
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
        return services::Status();
    }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
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
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input;            /*!< %Input data structure */

private:
    DistributedPartialResultStep1Ptr _partialResult;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__INIT__DISTRIBUTED_STEP2MASTER_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Provides methods for model-based training in the first step of distributed processing mode
 * <!-- \n<a href="DAAL-REF-GBT__REGRESSION__TRAINING__INIT-ALGORITHM">gradient boosted trees algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for model-based training, double or float
 * \tparam method           gradient boosted trees training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref gbt::regression::interface1::Model "Model" class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step2Master, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::gbt::regression::init::DistributedInput<step2Master> InputType;
    typedef algorithms::gbt::regression::init::Parameter                     ParameterType;
    typedef algorithms::gbt::regression::init::DistributedPartialResultStep2 PartialResultType;

    /**
     * Constructs a gradient boosted trees training algorithm
     */
    Distributed(size_t _maxBins = 256, size_t _minBinSize = 5);

    /**
     * Constructs a gradient boosted trees training algorithm by copying input objects
     * and parameters of another gradient boosted trees training algorithm in the first step of distributed processing mode
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step2Master, algorithmFPType, method> &other);

    ~Distributed()
    {
        delete _par;
    }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType& parameter() { return *static_cast<ParameterType*>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType& parameter() const { return *static_cast<const ParameterType*>(_par); }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    DistributedPartialResultStep2Ptr getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep2Ptr& partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated gradient boosted trees training algorithm with a copy of input objects
     * and parameters of this gradient boosted trees training algorithm algorithm
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
        return services::Status();
    }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
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
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input;            /*!< %Input data structure */

private:
    DistributedPartialResultStep2Ptr _partialResult;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__INIT__DISTRIBUTED_STEP3LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Provides methods for model-based training in the first step of distributed processing mode
 * <!-- \n<a href="DAAL-REF-GBT__REGRESSION__TRAINING__INIT-ALGORITHM">gradient boosted trees algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for model-based training, double or float
 * \tparam method           gradient boosted trees training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref gbt::regression::interface1::Model "Model" class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step3Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::gbt::regression::init::DistributedInput<step3Local>  InputType;
    typedef algorithms::gbt::regression::init::Parameter                     ParameterType;
    typedef algorithms::gbt::regression::init::DistributedPartialResultStep3 PartialResultType;

    /**
     * Constructs a gradient boosted trees training algorithm
     */
    Distributed(size_t _maxBins = 256);

    /**
     * Constructs a gradient boosted trees training algorithm by copying input objects
     * and parameters of another gradient boosted trees training algorithm in the first step of distributed processing mode
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step3Local, algorithmFPType, method> &other);

    ~Distributed()
    {
        delete _par;
    }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType& parameter() { return *static_cast<ParameterType*>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType& parameter() const { return *static_cast<const ParameterType*>(_par); }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    DistributedPartialResultStep3Ptr getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep3Ptr& partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated gradient boosted trees training algorithm with a copy of input objects
     * and parameters of this gradient boosted trees training algorithm algorithm
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

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        return services::Status();
    }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
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
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step3Local, algorithmFPType, method)(&_env);
        _in  = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input;            /*!< %Input data structure */

private:
    DistributedPartialResultStep3Ptr _partialResult;
};
} // namespace interface1
using interface1::DistributedContainer;
using interface1::Distributed;
} // namespace init
/** @} */
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif // __GBT_REGRESSION_TRAINING_INIT_DISTRIBUTED_H__
