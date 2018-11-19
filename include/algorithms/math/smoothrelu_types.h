/* file: smoothrelu_types.h */
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
//  Implementation of SmoothReLU algorithm interface.
//--
*/

#ifndef __SMOOTHRELU_TYPES_H__
#define __SMOOTHRELU_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for computing math functions
 */
namespace math
{
/**
 * @defgroup smoothrelu Smooth Rectifier Linear Unit(SmoothReLU)
 * \copydoc daal::algorithms::math::smoothrelu
 * @ingroup math
 * @{
 */
/**
 * \brief Contains classes for computing smooth rectified linear unit
 */
namespace smoothrelu
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__MATH__SMOOTHRELU__METHOD"></a>
 * Available methods for computing math functions
 */
enum Method
{
    defaultDense = 0       /*!< Default: performance-oriented method. */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__MATH__SMOOTHRELU__INPUTID"></a>
 * Available identifiers of input objects for the SmoothReLU algorithm
 */
enum InputId
{
    data,                /*!< %Input data table */
    lastInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__MATH__SMOOTHRELU__RESULTID"></a>
 * Available identifiers of results for the SmoothReLU algorithm
 */
enum ResultId
{
    value,      /*!< Table to store the result. Contains the lower triangle matrix L of the */
    lastResultId = value
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MATH__SMOOTHRELU__INPUT"></a>
 * \brief %Input parameters for the SmoothReLU algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input();

    /** Copy constructor */
    Input(const Input& other) : daal::algorithms::Input(other){}

    virtual ~Input() {}

    /**
     * Returns input NumericTable of the SmoothReLU algorithm
     * \param[in] id    Identifier of the input numeric table
     * \return          %Input numeric table that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets input for the SmoothReLU algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks parameters of the SmoothReLU algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
     *
     * \return Status of computation
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MATH__SMOOTHRELU__RESULT"></a>
 * \brief Results obtained with the compute() method of the SmoothReLU algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of math functions
     * \param[in] input  Pointer to the input structure
     * \param[in] par    Pointer to the parameter structure
     * \param[in] method Computation method of the algorithm
     *
     * \return Status of computation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

    /**
     * Returns result of the SmoothReLU algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of the SmoothReLU algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks the result of the smoothrelu algorithm
     * \param[in] in   %Input of algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
     *
     * \return Status of computation
     */
    services::Status check(const daal::algorithms::Input *in, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

/** @} */
} // namespace interface1
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

} // namespace smoothrelu
} // namespace math
} // namespace algorithm
} // namespace daal
#endif
