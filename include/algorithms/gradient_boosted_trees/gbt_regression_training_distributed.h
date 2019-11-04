/* file: gbt_regression_training_distributed.h */
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
//  Implementation of the interface for model-based training
//  in the distributed processing mode
//--
*/

#ifndef __GBT_REGRESSION_TRAINING_DISTRIBUTED_H__
#define __GBT_REGRESSION_TRAINING_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_training_types.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_model.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace training
{
namespace interface1
{
/**
 * @defgroup gbt_regression_training_distributed Distributed
 * @ingroup gbt_regression_training
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDCONTAINER"></a>
 * \brief Class containing computation methods for the gradient boosted trees regression
 *        model-based training using algorithmFPType precision arithmetic in the first step of the distributed processing mode
 */
template<ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the gradient boosted trees regression
 *        model-based training using algorithmFPType precision arithmetic in the first step of the distributed processing mode
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
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDCONTAINER_STEP2LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the gradient boosted trees regression
 *        model-based training using algorithmFPType precision arithmetic in the second step of the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step2Local, algorithmFPType, method, cpu> : public
    TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for model-based training with a specified environment
     * in the second step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of model-based training
     * in the second step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of model-based training
     * in the second step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDCONTAINER_STEP3LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the gradient boosted trees regression
 *        model-based training using algorithmFPType precision arithmetic in the third step of the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step3Local, algorithmFPType, method, cpu> : public
    TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for model-based training with a specified environment
     * in the third step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of model-based training
     * in the third step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of model-based training
     * in the third step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDCONTAINER_STEP4LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the gradient boosted trees regression
 *        model-based training using algorithmFPType precision arithmetic in the fourth step of the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step4Local, algorithmFPType, method, cpu> : public
    TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for model-based training with a specified environment
     * in the fourth step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of model-based training
     * in the fourth step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of model-based training
     * in the fourth step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDCONTAINER_STEP5LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the gradient boosted trees regression
 *        model-based training using algorithmFPType precision arithmetic in the fifth step of the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step5Local, algorithmFPType, method, cpu> : public
    TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for model-based training with a specified environment
     * in the fifth step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of model-based training
     * in the fifth step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of model-based training
     * in the fifth step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDCONTAINER_STEP6LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the gradient boosted trees regression
 *        model-based training using algorithmFPType precision arithmetic in the sixth step of the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step6Local, algorithmFPType, method, cpu> : public
    TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for model-based training with a specified environment
     * in the sixth step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of model-based training
     * in the sixth step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of model-based training
     * in the sixth step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTED"></a>
 * \brief Computes the results of the model-based training in the distributed processing mode
 * <!-- \n<a href="DAAL-REF--GBT__REGRESSION__TRAINING-ALGORITHM">gradient boosted trees algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for model-based training, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods
 *
 * \par References
 *      - \ref gbt::regression::interface1::Model "Model" class
 */
template<ComputeStep step, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Distributed {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Provides methods for model-based training in the first step of distributed processing mode
 * <!-- \n<a href="DAAL-REF-GBT__REGRESSION__TRAINING-ALGORITHM">gradient boosted trees algorithm description and usage models</a> -->
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
    typedef algorithms::gbt::regression::training::DistributedInput<step1Local>  InputType;
    typedef algorithms::gbt::regression::training::Parameter                     ParameterType;
    typedef algorithms::gbt::regression::training::DistributedPartialResultStep1 PartialResultType;

    /**
     * Constructs a gradient boosted trees training algorithm
     */
    Distributed();

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
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTED_STEP2LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Provides methods for model-based training in the second step of distributed processing mode
 * <!-- \n<a href="DAAL-REF-GBT__REGRESSION__TRAINING-ALGORITHM">gradient boosted trees algorithm description and usage models</a> -->
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
class DAAL_EXPORT Distributed<step2Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::gbt::regression::training::DistributedInput<step2Local>  InputType;
    typedef algorithms::gbt::regression::training::Parameter                     ParameterType;
    typedef algorithms::gbt::regression::training::DistributedPartialResultStep2 PartialResultType;

    /**
     * Constructs a gradient boosted trees training algorithm
     */
    Distributed();

    /**
     * Constructs a gradient boosted trees training algorithm by copying input objects
     * and parameters of another gradient boosted trees training algorithm in the second step of distributed processing mode
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step2Local, algorithmFPType, method> &other);

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
    services::SharedPtr<Distributed<step2Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step2Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step2Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step2Local, algorithmFPType, method>(*this);
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
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Local, algorithmFPType, method)(&_env);
        _in  = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input;            /*!< %Input data structure */

private:
    DistributedPartialResultStep2Ptr _partialResult;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTED_STEP3LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Provides methods for model-based training in the third step of distributed processing mode
 * <!-- \n<a href="DAAL-REF-GBT__REGRESSION__TRAINING-ALGORITHM">gradient boosted trees algorithm description and usage models</a> -->
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
    typedef algorithms::gbt::regression::training::DistributedInput<step3Local>  InputType;
    typedef algorithms::gbt::regression::training::Parameter                     ParameterType;
    typedef algorithms::gbt::regression::training::DistributedPartialResultStep3 PartialResultType;

    /**
     * Constructs a gradient boosted trees training algorithm
     */
    Distributed();

    /**
     * Constructs a gradient boosted trees training algorithm by copying input objects
     * and parameters of another gradient boosted trees training algorithm in the third step of distributed processing mode
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

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTED_STEP4LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Provides methods for model-based training in the fourth step of distributed processing mode
 * <!-- \n<a href="DAAL-REF-GBT__REGRESSION__TRAINING-ALGORITHM">gradient boosted trees algorithm description and usage models</a> -->
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
class DAAL_EXPORT Distributed<step4Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::gbt::regression::training::DistributedInput<step4Local>  InputType;
    typedef algorithms::gbt::regression::training::Parameter                     ParameterType;
    typedef algorithms::gbt::regression::training::DistributedPartialResultStep4 PartialResultType;

    /**
     * Constructs a gradient boosted trees training algorithm
     */
    Distributed();

    /**
     * Constructs a gradient boosted trees training algorithm by copying input objects
     * and parameters of another gradient boosted trees training algorithm in the fourth step of distributed processing mode
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step4Local, algorithmFPType, method> &other);

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
    DistributedPartialResultStep4Ptr getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep4Ptr& partialRes)
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
    services::SharedPtr<Distributed<step4Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step4Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step4Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step4Local, algorithmFPType, method>(*this);
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
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step4Local, algorithmFPType, method)(&_env);
        _in  = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input;            /*!< %Input data structure */

private:
    DistributedPartialResultStep4Ptr _partialResult;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTED_STEP5LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Provides methods for model-based training in the fifth step of distributed processing mode
 * <!-- \n<a href="DAAL-REF-GBT__REGRESSION__TRAINING-ALGORITHM">gradient boosted trees algorithm description and usage models</a> -->
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
class DAAL_EXPORT Distributed<step5Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::gbt::regression::training::DistributedInput<step5Local>  InputType;
    typedef algorithms::gbt::regression::training::Parameter                     ParameterType;
    typedef algorithms::gbt::regression::training::DistributedPartialResultStep5 PartialResultType;

    /**
     * Constructs a gradient boosted trees training algorithm
     */
    Distributed();

    /**
     * Constructs a gradient boosted trees training algorithm by copying input objects
     * and parameters of another gradient boosted trees training algorithm in the fifth step of distributed processing mode
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step5Local, algorithmFPType, method> &other);

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
    DistributedPartialResultStep5Ptr getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep5Ptr& partialRes)
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
    services::SharedPtr<Distributed<step5Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step5Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step5Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step5Local, algorithmFPType, method>(*this);
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
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step5Local, algorithmFPType, method)(&_env);
        _in  = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input;            /*!< %Input data structure */

private:
    DistributedPartialResultStep5Ptr _partialResult;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTED_STEP6LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Provides methods for model-based training in the sixth step of distributed processing mode
 * <!-- \n<a href="DAAL-REF-GBT__REGRESSION__TRAINING-ALGORITHM">gradient boosted trees algorithm description and usage models</a> -->
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
class DAAL_EXPORT Distributed<step6Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::gbt::regression::training::DistributedInput<step6Local>  InputType;
    typedef algorithms::gbt::regression::training::Parameter                     ParameterType;
    typedef algorithms::gbt::regression::training::DistributedPartialResultStep6 PartialResultType;

    /**
     * Constructs a gradient boosted trees training algorithm
     */
    Distributed();

    /**
     * Constructs a gradient boosted trees training algorithm by copying input objects
     * and parameters of another gradient boosted trees training algorithm in the sixth step of distributed processing mode
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step6Local, algorithmFPType, method> &other);

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
    DistributedPartialResultStep6Ptr getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep6Ptr& partialRes)
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
    services::SharedPtr<Distributed<step6Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step6Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step6Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step6Local, algorithmFPType, method>(*this);
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
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step6Local, algorithmFPType, method)(&_env);
        _in  = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input;            /*!< %Input data structure */

private:
    DistributedPartialResultStep6Ptr _partialResult;
};
/** @} */
} // namespace interface1
using interface1::DistributedContainer;
using interface1::Distributed;
}
}
}
}
}
#endif
