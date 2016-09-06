/* file: Batch.java */
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

package com.intel.daal.algorithms.kernel_function;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__BATCH"></a>
 * \brief Computes the kernel function in the batch processing mode
 * \n<a href="DAAL-REF-KERNEL_FUNCTION-ALGORITHM">Kernel function algorithm description and usage models</a>
 *
 * \par References
 *      - Parameter class
 */
public abstract class Batch extends AnalysisBatch {
    protected Batch(DaalContext context) {
        super(context);
    }

    /**
     * Returns the newly allocated kernel function algorithm with a copy of input objects
     * and parameters of this kernel function algorithm
     * @param context  Context to manage the kernel function
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract Batch clone(DaalContext context);
}
