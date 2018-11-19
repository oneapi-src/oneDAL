/* file: sum_of_functions_batch.h */
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
//  Implementation of the Sum of functions types.
//--
*/

#ifndef __SUM_OF_FUNCTIONS_BATCH_H__
#define __SUM_OF_FUNCTIONS_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "objective_function_batch.h"
#include "sum_of_functions_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sum_of_functions
{

namespace interface1
{
/**
 * @defgroup sum_of_functions_batch Batch
 * @ingroup sum_of_functions
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SUM_OF_FUNCTIONS__BATCH"></a>
 * \brief Interface for computing the Sum of functions in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-SUM_OF_FUNCTIONS-ALGORITHM">the Sum of functions description and usage models</a> -->
 *
 * \par Enumerations
 *      - \ref InputId  Identifiers of input objects for the Sum of functions
 *      - \ref objective_function::ResultId Result identifiers for the Sum of functions
 *
 * \par References
 *      - \ref interface1::Input class
 *      - \ref interface1::Result class
 */
class DAAL_EXPORT Batch : public objective_function::Batch
{
public:
    typedef objective_function::Batch super;

    typedef algorithms::optimization_solver::sum_of_functions::Input     InputType;
    typedef algorithms::optimization_solver::sum_of_functions::Parameter ParameterType;
    typedef super::ResultType                                            ResultType;

    /**
     *  Main constructor
     */
    Batch(size_t numberOfTerms, InputType *sumOfFunctionsInput, ParameterType *sumOfFunctionsParameter) :
        sumOfFunctionsInput(sumOfFunctionsInput),
        sumOfFunctionsParameter(sumOfFunctionsParameter)
    {
        initialize();
        if(sumOfFunctionsParameter != NULL) {sumOfFunctionsParameter->numberOfTerms = numberOfTerms;}
    }

    /**
     * Constructs the Sum of functions by copying input objects and parameters
     * of another the Sum of functions
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch &other) : sumOfFunctionsInput(other.sumOfFunctionsInput),
        sumOfFunctionsParameter(other.sumOfFunctionsParameter)
    {
        initialize();
    }

    virtual ~Batch() {}

    /**
     * Returns a pointer to the newly allocated Sum of functions with a copy of input objects
     * of this Sum of functions
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch> clone() const
    {
        return services::SharedPtr<Batch>(cloneImpl());
    }

    ParameterType *sumOfFunctionsParameter; /*!< Pointer to the parameter to use one object in inherited class */
    InputType *sumOfFunctionsInput;         /*!< Pointer to the input to use one object in inherited class */

protected:
    virtual Batch *cloneImpl() const DAAL_C11_OVERRIDE = 0;

    void initialize()
    {}
};
typedef services::SharedPtr<Batch> BatchPtr;

/** @} */
} // namespace interface1
using interface1::Batch;
using interface1::BatchPtr;

} // namespace sum_of_functions
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
