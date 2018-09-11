/* file: OptionalInputId.java */
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

/**
 * @ingroup iterative_solver
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.iterative_solver;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__OPTIONALINPUTID"></a>
 * @brief Available identifiers of optional input objects for the iterative algorithm
 */
public final class OptionalInputId {
    private int _value;

    /**
     * Constructs the optional input object identifier using the provided value
     * @param value     Value corresponding to the optional input object identifier
     */
    public OptionalInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the optional input object identifier
     * @return Value corresponding to the optional input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int optionalArgumentId = 1;

    public static final OptionalInputId optionalArgument = new OptionalInputId(optionalArgumentId); /*!< %Algorithm-specific input data */
}
/** @} */
