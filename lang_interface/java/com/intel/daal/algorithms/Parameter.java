/* file: Parameter.java */
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
     * @brief Parameter default constructor
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
