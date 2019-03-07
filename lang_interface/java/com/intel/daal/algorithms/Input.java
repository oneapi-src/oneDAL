/* file: Input.java */
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

import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__INPUT"></a>
 *  \brief Base class to represent computation input arguments.
 *         Algorithm-specific input arguments are represented as derivative classes of the Input class.
 */
abstract public class Input extends ContextClient {
    /**
     * @brief Pointer to C++ implementation of the parameter.
     */
    protected long cObject;

    protected Input(DaalContext context) {
        super(context);
    }

    /**
     * Constructs parameter from C++ parameter
     * @param context Context to manage Input object
     * @param cInput  Address of C++ parameter
     */
    public Input(DaalContext context, long cInput) {
        super(context);
        this.cObject = cInput;
    }

    public long getCObject() {
        return this.cObject;
    }

    /**
     * Checks the correctness of the Input object
     * @param parameter     Parameters of the algorithm
     * @param method        Computation method
     */
    public void check(Parameter parameter, int method) {
        cCheck(this.cObject, parameter.getCObject(), method);
    }

    /**
     * Releases memory allocated for the native parameter object
     */
    @Override
    public void dispose() {
        /* memory will be freed by algorithm object */ }

    private native void cCheck(long inputAddr, long parAddr, int method);
}
/** @} */
