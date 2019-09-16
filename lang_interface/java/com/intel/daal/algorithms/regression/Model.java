/* file: Model.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 * @defgroup regressor Base Regressor
 * @brief Contains classes for working with regressors
 * @ingroup regression
 */
package com.intel.daal.algorithms.regression;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__REGRESSION__MODEL"></a>
 * @brief Base class for models of the regression algorithms.
 */
public class Model extends com.intel.daal.algorithms.Model {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the model of the regression algorithm
     * @param context   Context to manage the model of the regression algorithm
     */
    public Model(DaalContext context) {
        super(context);
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
    }
}
/** @} */
