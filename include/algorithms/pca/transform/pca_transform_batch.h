/* file: pca_transform_batch.h */
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
//  Implementation of the interface for the PCA transformation algorithm in the
//  batch processing mode
//--
*/

#ifndef __PCA_TRANSFORM_BATCH_H__
#define __PCA_TRANSFORM_BATCH_H__

#include "algorithms/algorithm.h"
#include "services/daal_defines.h"
#include "algorithms/pca/transform/pca_transform_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace transform
{
namespace interface1
{
/**
 * @defgroup pca_transform_batch Batch
 * @ingroup pca_transform
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__TRANSFORM__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the PCA transformation algorithm in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the PCA transformation algorithm, double or float
 * \tparam method           Computation method of the PCA transformation algorithm, \ref daal::algorithms::pca::transform::Method
 *
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
    * Constructs a container for the PCA transformation algorithm with a specified environment
    * in the batch processing mode
    * \param[in] daalEnv   Environment object
    */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
    * Computes the result of the PCA transformation algorithm in the batch processing mode
    */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__PCA__TRANSFORM__BATCH"></a>
* \brief Computes the results of the PCA transformation algorithm in the batch processing mode.
* <!-- \n<a href="DAAL-REF-PCA-ALGORITHM">PCA transformation algorithm description and usage models</a> -->
*
* \tparam algorithmFPType  Data type to use in intermediate computations for the PCA transformation algorithm, double or float
* \tparam method           Computation method of the algorithm, \ref daal::algorithms::pca::transform::Method
*
* \par Enumerations
*      - \ref Method   Computation methods for the PCA transformation algorithm
*/
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::pca::transform::Input     InputType;
    typedef algorithms::pca::transform::Parameter ParameterType;
    typedef algorithms::pca::transform::Result    ResultType;

    InputType input;            /*!< Input object */
    ParameterType parameter;    /*!< PCA transformation parameters */

    /**
    * Constructs a PCA transformation algorithm
    * \param[in] nComponents Number of principal components
    */
    Batch(size_t nComponents = 0): parameter(nComponents)
    {
        initialize();
    }

    /**
    * Constructs a PCA transformation algorithm by copying input objects and parameters
    * of another PCA transformation algorithm
    * \param[in] other An algorithm to be used as the source to initialize the input objects
    *                  and parameters of the algorithm
    */
    Batch(const Batch<algorithmFPType, method> &other) : input(other.input), parameter(other.parameter)
    {
        initialize();
    }

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
    * Returns the structure that contains the results of the PCA transformation algorithm
    * \return Structure that contains the results of the PCA transformation algorithm
    */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
    * Register user-allocated memory to store the results of the PCA transformation algorithm
    * \return Structure to store the results of the PCA transformation algorithm
    */
    services::Status setResult(const ResultPtr& res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
            _result = res;
        _res = _result.get();
        return services::Status();
    }

    /**
    * Returns a pointer to the newly allocated PCA transformation algorithm
    * with a copy of input objects and parameters of this PCA transformation algorithm
    * \return Pointer to the newly allocated algorithm
    */
    services::SharedPtr<daal::algorithms::pca::transform::interface1::Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, 0);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
    }

private:
    ResultPtr _result;
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace transform
} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
