/* file: PartialResult.java */
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
 * @ingroup base_algorithms
 * @{
 */
package com.intel.daal.algorithms;

import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__PARTIALRESULT"></a>
 *  \brief %Base class to represent partial results of the computation.
 *         Algorithm-specific partial results are represented as derivative classes of the PartialResult class.
 */
abstract public class PartialResult extends SerializableBase {

    /**
     * Constructs the partial result of the algorithm
     * @param context       Context to manage the partial result of the algorithm
     */
    protected PartialResult(DaalContext context) {
        super(context);
    }

    /**
     * Constructs parameter from C++ parameter
     * @param context Context to manage the partial result
     * @param cPartialResult Address of C++ parameter
     */
    public PartialResult(DaalContext context, long cPartialResult) {
        super(context);
        this.cObject = cPartialResult;
    }

    /**
     * Checks the correctness of the partial results
     * @param input         Input object
     * @param parameter     Parameters of the algorithm
     * @param method        Computation method
     */
    public void check(Input input, Parameter parameter, int method) {
        cCheckInput(this.cObject, input.getCObject(), parameter.getCObject(), method);
    }

    /**
     * Checks the correctness of the partial result
     * @param parameter     Parameters of the algorithm
     * @param method        Computation method
     */
    public void check(Parameter parameter, int method) {
        cCheck(this.cObject, parameter.getCObject(), method);
    }

    private native void cCheckInput(long partResAddr, long inputAddr, long parAddr, int method);

    private native void cCheck(long partResAddr, long parAddr, int method);
}
/** @} */
