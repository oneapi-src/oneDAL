/* file: Parameter.java */
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
 *  <a name="DAAL-CLASS-ALGORITHMS__PARAMETER"></a>
 *  \brief %Base class to represent computation parameters.
 *         Algorithm-specific parameters are represented as derivative classes of the Parameter class.
 */
abstract public class Parameter extends ContextClient {
    /**
     * @brief Pointer to C++ implementation of the parameter.
     */
    public long cObject;

    /**
     * Constructs the parameter for the algorithm
     * @param context       Context to manage the parameter for the algorithm
     */
    protected Parameter(DaalContext context) {
        super(context);
        this.cObject = 0;
    }

    /**
     * Constructs parameter from C++ parameter
     * @param context Context to manage the parameter
     * @param cParameter Address of C++ parameter
     */
    public Parameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    public long getCObject() {
        return this.cObject;
    }

    /**
     * Checks the correctness of the algorithm parameters
     */
    public void check() {
        cCheck(this.cObject);
    }

    /**
     * Releases memory allocated for the native parameter object
     */
    @Override
    public void dispose() {
        /* memory will be freed by algorithm object */
    }

    private native void cDispose(long parAddr);

    private native void cCheck(long parAddr);
}
/** @} */
