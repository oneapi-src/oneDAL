/* file: pca_online.h */
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
//  Implementation of the interface for the PCA algorithm in the online processing mode
//--
*/

#ifndef __PCA_ONLINE_H__
#define __PCA_ONLINE_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "algorithms/pca/pca_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{

namespace interface1
{
/** @defgroup pca_online Online
 * @ingroup pca
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINECONTAINER"></a>
 * \brief Class containing methods to compute the result of the PCA algorithm
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT OnlineContainer : public AnalysisContainerIface<online> {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINECONTAINER_ALGORITHMFPTYPE_CORRELATIONDENSE_CPU"></a>
 * \brief Class containing methods to compute the result of the PCA algorithm
 */
template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT OnlineContainer<algorithmFPType, correlationDense, cpu> : public AnalysisContainerIface<online>
{
public:
    /**
     * Constructs a container for the PCA algorithm with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~OnlineContainer();

    /**
     * Computes a partial result of the PCA algorithm in the online processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the PCA algorithm in the online processing mode
     */
    void finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINECONTAINER_ALGORITHMFPTYPE_SVDDENSE_CPU"></a>
 * \brief Class containing methods to compute the results of the PCA algorithm
 */
template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT OnlineContainer<algorithmFPType, svdDense, cpu> : public AnalysisContainerIface<online>
{
public:
    /**
     * Constructs a container for the PCA algorithm with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~OnlineContainer();

    /**
     * Computes a partial result of the PCA algorithm in the online processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the PCA algorithm in the online processing mode
     */
    void finalizeCompute() DAAL_C11_OVERRIDE;
};


template<typename algorithmFPType = double, Method method = correlationDense>
class DAAL_EXPORT Online : public Analysis<online> {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINE_ALGORITHMFPTYPE_CORRELATIONDENSE"></a>
 * \brief Computes the results of the PCA Correlation algorithm
 * \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the PCA algorithm, double or float
 *
 * \par References
 *      - \ref interface1::OnlineParameter class
 */
template<typename algorithmFPType>
class DAAL_EXPORT Online<algorithmFPType, correlationDense> : public Analysis<online>
{
public:
    /** Default constructor */
    Online()
    {
        initialize();
    }

    /**
     * Constructs a PCA algorithm by copying input objects and parameters of another PCA algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Online(const Online<algorithmFPType, correlationDense> &other)
    {
        initialize();
        input.set(data, other.input.get(data));
        parameter = other.parameter;
    }

    ~Online() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    int getMethod() const DAAL_C11_OVERRIDE { return(int)correlationDense; }

    /**
     * Registers user-allocated  memory to store the results of the PCA algorithm
     * \param[in] partialResult    Structure for storing partial result of the PCA algorithm
     */
    void setPartialResult(const services::SharedPtr<PartialResultImpl<algorithmFPType, correlationDense> >& partialResult)
    {
        _partialResult = partialResult;
        _pres = _partialResult.get();
    }

    /**
     * Registers user-allocated memory to store the results of the PCA algorithm
     * \param[in] res    Structure to store the results of the PCA algorithm
     */
    void setResult(const services::SharedPtr<Result>& res)
    {
        DAAL_CHECK(res, ErrorNullResult)
        _result = res;
        _res = _result.get();
    }

    /**
     * Returns the structure that contains partial results of the PCA algorithm
     * \return Structure that contains partial results of the PCA algorithm
     */
    services::SharedPtr<PartialResultImpl<algorithmFPType, correlationDense> > getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Returns structure that contains the results of the PCA algorithm
     * \return Structure that contains the results of the PCA algorithm
     */
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Returns a pointer to the newly allocated PCA algorithm
     * with a copy of input objects and parameters of this PCA algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Online<algorithmFPType, correlationDense> > clone() const
    {
        return services::SharedPtr<Online<algorithmFPType, correlationDense> >(cloneImpl());
    }

    Input input; /*!< Input data structure */
    OnlineParameter<algorithmFPType, correlationDense> parameter; /*!< Parameters */

protected:
    services::SharedPtr<PartialResultImpl<algorithmFPType, correlationDense> > _partialResult;
    services::SharedPtr<Result> _result;

    virtual Online<algorithmFPType, correlationDense> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Online<algorithmFPType, correlationDense>(*this);
    }

    void allocateResult() DAAL_C11_OVERRIDE
    {
        _result->allocate<algorithmFPType>(_pres, &parameter, correlationDense);
        _res = _result.get();
    }

    void allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult->allocate(&input, &parameter, correlationDense);
        _pres = _partialResult.get();
    }

    void initializePartialResult() DAAL_C11_OVERRIDE
    {
        services::SharedPtr<PartialResult<correlationDense> > pres
        = services::staticPointerCast<PartialResult<correlationDense>, PartialResultImpl<algorithmFPType, correlationDense> >(_partialResult);
        (*parameter.initializationProcedure)(input, pres);
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(online, OnlineContainer, algorithmFPType, correlationDense)(&_env);
        _in = &input;
        _par = &parameter;
        _partialResult = services::SharedPtr<PartialResultImpl<algorithmFPType, correlationDense> >(
                new PartialResultImpl<algorithmFPType, correlationDense>());
        _result = services::SharedPtr<Result>(new Result());
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINE_ALGORITHMFPTYPE_SVDDENSE"></a>
 * \brief Computes the results of the PCA SVD algorithm
 * \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the PCA algorithm, double or float
 *
 * \par References
 *      - \ref interface1::OnlineParameter class
 */
template<typename algorithmFPType>
class DAAL_EXPORT Online<algorithmFPType, svdDense> : public Analysis<online>
{
public:
    /** Default constructor */
    Online()
    {
        initialize();
    }

    /**
     * Constructs a PCA algorithm by copying input objects and parameters of another PCA algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Online(const Online<algorithmFPType, svdDense> &other)
    {
        initialize();
        input.set(data, other.input.get(data));
        parameter = other.parameter;
    }

    ~Online() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    int getMethod() const DAAL_C11_OVERRIDE { return(int)svdDense; }

    /**
     * Registers user-allocated  memory to store the results of the PCA algorithm
     * \param[in] partialResult    Structure for storing partial result of the PCA algorithm
     */
    void setPartialResult(const services::SharedPtr<PartialResultImpl<algorithmFPType, svdDense> >& partialResult)
    {
        _partialResult = partialResult;
        _pres = _partialResult.get();
    }

    /**
     * Registers user-allocated memory to store the results of the PCA algorithm
     * \param[in] res    Structure to store the results of the PCA algorithm
     */
    void setResult(const services::SharedPtr<Result>& res)
    {
        DAAL_CHECK(res, ErrorNullResult)
        _result = res;
        _res = _result.get();
    }

    /**
     * Returns the structure that contains partial results of the PCA algorithm
     * \return Structure that contains partial results of the PCA algorithm
     */
    services::SharedPtr<PartialResultImpl<algorithmFPType, svdDense> > getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Returns structure that contains the results of the PCA algorithm
     * \return Structure that contains the results of the PCA algorithm
     */
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Returns a pointer to the newly allocated PCA algorithm
     * with a copy of input objects and parameters of this PCA algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Online<algorithmFPType, svdDense> > clone() const
    {
        return services::SharedPtr<Online<algorithmFPType, svdDense> >(cloneImpl());
    }

    Input input; /*!< Input data structure */
    OnlineParameter<algorithmFPType, svdDense> parameter; /*!< Parameters */

protected:
    services::SharedPtr<PartialResultImpl<algorithmFPType, svdDense> > _partialResult;
    services::SharedPtr<Result> _result;

    virtual Online<algorithmFPType, svdDense> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Online<algorithmFPType, svdDense>(*this);
    }

    void allocateResult() DAAL_C11_OVERRIDE
    {
        _result->allocate<algorithmFPType>(_pres, &parameter, svdDense);
        _res = _result.get();
    }

    void allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult->allocate(&input, &parameter, svdDense);
        _pres = _partialResult.get();
    }

    void initializePartialResult() DAAL_C11_OVERRIDE
    {
        services::SharedPtr<PartialResult<svdDense> > pres
        = services::staticPointerCast<PartialResult<svdDense>, PartialResultImpl<algorithmFPType, svdDense> >(_partialResult);
        (*parameter.initializationProcedure)(input, pres);
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(online, OnlineContainer, algorithmFPType, svdDense)(&_env);
        _in = &input;
        _par = &parameter;
        _partialResult = services::SharedPtr<PartialResultImpl<algorithmFPType, svdDense> >(
                new PartialResultImpl<algorithmFPType, svdDense>());
        _result = services::SharedPtr<Result>(new Result());
    }
};
/** @} */
} // namespace interface1
using interface1::OnlineContainer;
using interface1::Online;

}
}
}
#endif
