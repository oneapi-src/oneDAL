/* file: InitDistributedStep2MasterInput.java */
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

package com.intel.daal.algorithms.kmeans.init;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief Input objects for computing initial clusters for the K-Means algorithm.
 *        The class represents input objects for computing initial clusters for the algorithm on the master node.
 */
public final class InitDistributedStep2MasterInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public InitDistributedStep2MasterInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Adds partial results computed on local nodes to the input for computing initial clusters for the K-Means algorithm
     * in the second step in the distributed processing mode
     * @param id            Identifier of the input object
     * @param pres          Partial results of the K-Means initialization algorithm obtained in the
     *                      first step of the distributed processing mode
     */

    public void add(InitDistributedStep2MasterInputId id, InitPartialResult pres) {
        cAddInput(cObject, id.getValue(), pres.getCObject());
    }

    private native void cAddInput(long inputAddr, int id, long presAddr);
}
