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

package com.intel.daal.algorithms.multivariate_outlier_detection.defaultdense;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__DEFAULTDENSE__PARAMETER"></a>
 * @brief Parameters for the multivariate outlier detection compute() used with the defaultDense method
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Parameter(DaalContext context, long cParameter) {
        super(context);

        this.cObject = cParameter;
        _initializationProcedure = null;
    }

    /**
     * Set initialization procedure for specifying initial parameters of the multivariate outlier detection algorithm
     * @param initializationProcedure   Initialization procedure
     */
    public void setInitializationProcedure(Object initializationProcedure) {
        _initializationProcedure = initializationProcedure;
    }

    /**
     * Get initialization procedure for specifying initial parameters of the multivariate outlier detection algorithm
     * @return  Initialization procedure
     */
    public Object getInitializationProcedure() {
        return _initializationProcedure;
    }

    private Object _initializationProcedure;
}
