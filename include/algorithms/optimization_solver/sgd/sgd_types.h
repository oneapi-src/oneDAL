/* file: sgd_types.h */
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
//  Implementation of the Stochastic gradient descent algorithm types.
//--
*/

#ifndef __SGD_TYPES_H__
#define __SGD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/optimization_solver/iterative_solver/iterative_solver_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
/**
 * @defgroup sgd Stochastic Gradient Descent Algorithm
 * \copydoc daal::algorithms::optimization_solver::sgd
 * @ingroup optimization_solver
 * @{
 */
/**
 * \brief Contains classes for computing the Stochastic gradient descent
 */
namespace sgd
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__METHOD"></a>
 * Available methods for computing the Stochastic gradient descent
 */
enum Method
{
    defaultDense = 0, /*!< Default: Required gradient is computed using only one term of objective function */
    miniBatch = 1     /*!< Required gradient is computed using batchSize terms of objective function  */
};

/**
* For internal use only
*/
enum InternalOptionalDataId
{
    rngState, /*!< Memory block with random numbers generator state */
    optionalDataSize
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__BASEPARAMETER"></a>
 * \brief %BaseParameter base class for the Stochastic gradient descent algorithm
 *
 * \snippet optimization_solver/sgd/sgd_types.h BaseParameter source code
 */
/* [BaseParameter source code] */
struct DAAL_EXPORT BaseParameter : public optimization_solver::iterative_solver::Parameter
{
    /**
     * Constructs the parameter base class of the Stochastic gradient descent algorithm
     * \param[in] function             Objective function represented as sum of functions
     * \param[in] nIterations          Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function.
     *                                 If no indices are provided, the implementation will generate random indices.
     * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
     * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function.
     */
    BaseParameter(
        const sum_of_functions::BatchPtr& function,
        size_t nIterations = 100,
        double accuracyThreshold = 1.0e-05,
        data_management::NumericTablePtr batchIndices = data_management::NumericTablePtr(),
        data_management::NumericTablePtr learningRateSequence = data_management::NumericTablePtr(
                    new data_management::HomogenNumericTable<double>(
                        1, 1, data_management::NumericTableIface::doAllocate, 1.0)),
        size_t seed = 777 );

    virtual ~BaseParameter() {}

    /**
     * Checks the correctness of the parameter
     */
    virtual void check() const;

    data_management::NumericTablePtr batchIndices;         /*!< Numeric table that represents 32 bit integer indices of terms
                                                                in the objective function. If no indices are provided,
                                                                the implementation will generate random indices. */
    data_management::NumericTablePtr learningRateSequence; /*!< Numeric table that contains values of the learning rate sequence */
    size_t                           seed;                 /*!< Seed for random generation of 32 bit integer indices of terms
                                                                  in the objective function. */
};
/* [BaseParameter source code] */

template<Method method>
struct Parameter : public BaseParameter {};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__PARAMETER_DEFAULTDENSE"></a>
 * \brief %Parameter for the Stochastic gradient descent algorithm
 *
 * \snippet optimization_solver/sgd/sgd_types.h ParameterDefaultDense source code
 */
/* [ParameterDefaultDense source code] */
template<>
struct DAAL_EXPORT Parameter<defaultDense> : public BaseParameter
{
    /**
     * \param[in] function             Objective function represented as sum of functions
     * \param[in] nIterations          Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function. If no indices are
                                       provided, the implementation will generate random indices.
     * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
     * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function.
     */
    Parameter(
        const sum_of_functions::BatchPtr& function,
        size_t nIterations = 100,
        double accuracyThreshold = 1.0e-05,
        data_management::NumericTablePtr batchIndices = data_management::NumericTablePtr(),
        data_management::NumericTablePtr learningRateSequence = data_management::NumericTablePtr(
                    new data_management::HomogenNumericTable<double>(
                        1, 1, data_management::NumericTableIface::doAllocate, 1.0)),
        size_t seed = 777 );

    /**
     * Checks the correctness of the parameter
     */
    virtual void check() const;

    virtual ~Parameter() {}
};
/* [ParameterDefaultDense source code] */

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__PARAMETER_MINIBATCH"></a>
 * \brief %Parameter for the Stochastic gradient descent algorithm
 *
 * \snippet optimization_solver/sgd/sgd_types.h ParameterMiniBatch source code
 */
/* [ParameterMiniBatch source code] */
template<>
struct DAAL_EXPORT Parameter<miniBatch> : public BaseParameter
{
    /**
     * Constructs the parameter class of the Stochastic gradient descent algorithm
     * \param[in] function             Objective function represented as sum of functions
     * \param[in] nIterations          Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function. If no indices
                                       are provided, the implementation will generate random indices.
     * \param[in] batchSize            Number of batch indices to compute the stochastic gradient. If batchSize is equal to the number of terms
                                       in objective function then no random sampling is performed, and all terms are used to calculate the gradient.
                                       This parameter is ignored if batchIndices is provided.
     * \param[in] conservativeSequence Numeric table of values of the conservative coefficient sequence
     * \param[in] innerNIterations     Number of inner iterations
     * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
     * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function.
     */
    Parameter(
        const sum_of_functions::BatchPtr& function,
        size_t nIterations = 100,
        double accuracyThreshold = 1.0e-05,
        data_management::NumericTablePtr batchIndices = data_management::NumericTablePtr(),
        size_t batchSize = 128,
        data_management::NumericTablePtr conservativeSequence = data_management::NumericTablePtr(
                    new data_management::HomogenNumericTable<double>(
                        1, 1, data_management::NumericTableIface::doAllocate, 1.0)),
        size_t innerNIterations = 5,
        data_management::NumericTablePtr learningRateSequence = data_management::NumericTablePtr(
                    new data_management::HomogenNumericTable<double>(
                        1, 1, data_management::NumericTableIface::doAllocate, 1.0)),
        size_t seed = 777 );

    /**
     * Checks the correctness of the parameter
     */
    virtual void check() const;

    virtual ~Parameter() {}

    size_t                           batchSize;             /*!< Number of batch indices to compute the stochastic gradient.
                                                                If batchSize is equal to the number of terms in objective
                                                                function then no random sampling is performed, and all terms are
                                                                used to calculate the gradient. This parameter is ignored
                                                                if batchIndices is provided. */
    data_management::NumericTablePtr conservativeSequence; /*!< Numeric table of values of the conservative coefficient sequence */
    size_t                           innerNIterations;
};
/* [ParameterMiniBatch source code] */
/** @} */

/**
* <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__INPUT"></a>
* \brief %Input for the Stochastic gradient descent algorithm
*
* \snippet optimization_solver/sgd/sgd_types.h Input source code
*/
/* [Input source code] */
class DAAL_EXPORT Input : public optimization_solver::iterative_solver::Input
{
protected:
    typedef optimization_solver::iterative_solver::Input super;
public:
    /**
    * Checks the correctness of the input
    * \param[in] par       Pointer to the structure of the algorithm parameters
    * \param[in] method    Computation method
    */
    virtual void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};
/* [Input source code] */
/** @} */

/**
* <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__RESULT"></a>
* \brief Results obtained with the compute() method of the sgd algorithm in the batch processing mode
*/
/* [Result source code] */
class DAAL_EXPORT Result : public optimization_solver::iterative_solver::Result
{
protected:
    typedef optimization_solver::iterative_solver::Result super;

public:
    /**
    * Allocates memory to store the results of the iterative solver algorithm
    * \param[in] input  Pointer to the input structure
    * \param[in] par    Pointer to the parameter structure
    * \param[in] method Computation method of the algorithm
    */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

    /**
    * Checks the result of the iterative solver algorithm
    * \param[in] input   %Input of algorithm
    * \param[in] par     %Parameter of algorithm
    * \param[in] method  Computation method of the algorithm
    */
    virtual void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
        int method) const DAAL_C11_OVERRIDE;

    /** Returns a serialization tag, a unique identifier of this class used in serialization
    * \return Serialization tag
    */
    virtual int getSerializationTag() DAAL_C11_OVERRIDE{ return SERIALIZATION_SGD_RESULT_ID; }
};
/* [Result source code] */
/** @} */

} // namespace interface1
using interface1::BaseParameter;
using interface1::Parameter;
using interface1::Input;
using interface1::Result;

} // namespace sgd
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
