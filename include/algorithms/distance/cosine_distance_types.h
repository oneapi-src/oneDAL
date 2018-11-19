/* file: cosine_distance_types.h */
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
//  Implementation of cosine distance algorithm interface.
//--
*/

#ifndef __COSDISTANCE_TYPES_H__
#define __COSDISTANCE_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/symmetric_matrix.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup cosine_distance Cosine Distance Matrix
 * \copydoc daal::algorithms::cosine_distance
 * @ingroup analysis
 * @{
 */
/**
* \brief Contains classes for computing the cosine distance
*/
namespace cosine_distance
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__COSINE_DISTANCE__METHOD"></a>
 * Available methods for computing the cosine distance
 */
enum Method
{
    defaultDense = 0       /*!< Default: performance-oriented method. */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__COSINE_DISTANCE__INPUTID"></a>
 * Available identifiers of input objects for the cosine distance algorithm
 */
enum InputId
{
    data,            /*!< %Input data table */
    lastInputId = data
};
/**
 * <a name="DAAL-ENUM-ALGORITHMS__COSINE_DISTANCE__RESULTID"></a>
 * Available identifiers of results for the cosine distance algorithm
 */
enum ResultId
{
    cosineDistance,           /*!< Table to store the result.*/
    lastResultId = cosineDistance
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__COSINE_DISTANCE__INPUT"></a>
 * \brief %Input objects for the cosine distance algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input();
    Input(const Input& other) : daal::algorithms::Input(other){}

    virtual ~Input() {}

    /**
    * Returns the input object of the cosine distance algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(InputId id) const;

    /**
    * Sets the input object for the cosine distance algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(InputId id, const data_management::NumericTablePtr &ptr);

    /**
    * Checks the parameters of the cosine distance algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  computation method
    */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COSINE_DISTANCE__RESULT"></a>
 * \brief Results obtained with the compute() method of the cosine distance algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory to store results of the cosine distance algorithm
     * \param[in] input  Pointer to input structure
     * \param[in] par    Pointer to parameter structure
     * \param[in] method Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

    /**
     * Returns the result of the cosine distance algorithm
     * \param[in] id   Identifier of the result
     * \return         %Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of the cosine distance algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the result object
     */
    void set(ResultId id, const data_management::NumericTablePtr &ptr);

    /**
    * Checks the result of the cosine distance algorithm
    * \param[in] input   %Input of the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method
    */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

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

} // namespace cosine_distance
} // namespace algorithms
} // namespace daal
#endif
