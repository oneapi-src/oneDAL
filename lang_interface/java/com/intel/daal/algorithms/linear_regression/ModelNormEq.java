/* file: ModelNormEq.java */
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

package com.intel.daal.algorithms.linear_regression;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__MODELNORMEQ"></a>
 * @brief %Model trained by the linear regression algorithm using the normal equations method
 *
 */
public class ModelNormEq extends Model {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public ModelNormEq(DaalContext context, long nFeatures, long nResponses, Parameter parameter, Class<?> cls) {
        super(context, 0);
        if (cls == Double.class) {
            this.cObject = cInitDouble(nFeatures, nResponses, parameter.cObject);
        } else if (cls == Float.class) {
            this.cObject = cInitFloat(nFeatures, nResponses, parameter.cObject);
        } else {
            throw new IllegalArgumentException("type unsupported");
        }
    }

    public ModelNormEq(DaalContext context, long cModel) {
        super(context, cModel);
    }

    private native long cInitDouble(long nFeatures, long nResponses, long cObject);

    private native long cInitFloat(long nFeatures, long nResponses, long cObject);
}
