/* file: InitDistributedStep1LocalInput.java */
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
 * @ingroup implicit_als_init_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training.init;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITDISTRIBUTEDSTEP1LOCALINPUT"></a>
 * @brief %Input objects for the implicit ALS initialization algorithm in the second step
 *        of the distributed processing mode
 */

public final class InitDistributedStep1LocalInput extends InitInput {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public InitDistributedStep1LocalInput(DaalContext context, long cAlgorithm, Precision prec, InitMethod method) {
        super(context);
        this.cObject = cGetInput(cAlgorithm, prec.getValue(), method.getValue());
    }

    private native long cGetInput(long cAlgorithm, int prec, int method);
}
/** @} */
