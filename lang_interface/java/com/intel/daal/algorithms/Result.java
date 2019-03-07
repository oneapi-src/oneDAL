/* file: Result.java */
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

/**
 * @ingroup base_algorithms
 * @{
 */
package com.intel.daal.algorithms;

import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__RESULT"></a>
 *  \brief %Base class to represent final results of the computation.
 *         Algorithm-specific final results are represented as derivative classes of the Result class.
 */
abstract public class Result extends SerializableBase {

    /**
     * Constructs the result of the algorithm
     * @param context   Context to manage the result of the algorithm
     */
    protected Result(DaalContext context) {
        super(context);
    }

    /**
     * Constructs parameter from C++ parameter
     * @param context Context to manage the result
     * @param cResult Address of C++ parameter
     */
    public Result(DaalContext context, long cResult) {
        super(context);
        this.cObject = cResult;
    }

    /**
     * Checks the correctness of the result
     * @param input         Input object
     * @param parameter     Parameters of the algorithm
     * @param method        Computation method
     */
    public void check(Input input, Parameter parameter, int method) {
        cCheckInput(this.cObject, input.getCObject(), parameter.getCObject(), method);
    }

    /**
     * Checks the correctness of the result
     * @param partialResult Partial result of the algorithm
     * @param parameter     Parameters of the algorithm
     * @param method        Computation method
     */
    public void check(PartialResult partialResult, Parameter parameter, int method) {
        cCheckPartRes(this.cObject, partialResult.getCObject(), parameter.getCObject(), method);
    }

    private native void cDispose(long parAddr);

    private native void cCheckInput(long resAddr, long inputAddr, long parAddr, int method);

    private native void cCheckPartRes(long resAddr, long partResAddr, long parAddr, int method);

}
/** @} */
