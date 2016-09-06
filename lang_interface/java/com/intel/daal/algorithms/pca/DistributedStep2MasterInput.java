/* file: DistributedStep2MasterInput.java */
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

package com.intel.daal.algorithms.pca;

import com.intel.daal.algorithms.PartialResult;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief Input objects for the second step of the PCA algorithm
 *        in the distributed processing mode.
 */
public final class DistributedStep2MasterInput extends com.intel.daal.algorithms.Input {
    public Method                               method;  /*!< Computation method for the algorithm */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public DistributedStep2MasterInput(DaalContext context, long cObject, Method method) {
        super(context, cObject);
        this.method = method;
    }

    /**
     * Adds partial result to the input of the PCA algorithm on the second step in the distributed processing mode
     * @param id            Identifier of the input object
     * @param pres          Partial result obtained on the first step of the PCA algorithm in the distributed processing mode
     */
    public void add(MasterInputId id, PartialResult pres) {
        cAddInput(cObject, id.getValue(), pres.getCObject(), method.getValue());
    }

    private native void cAddInput(long algAddr, int id, long presAddr, int method);
}
