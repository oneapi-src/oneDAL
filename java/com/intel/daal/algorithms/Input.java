/* file: Input.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
