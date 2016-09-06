/* file: CompressionParameter.java */
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

package com.intel.daal.data_management.compression;

import com.intel.daal.algorithms.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__COMPRESSIONPARAMETER"></a>
 * @brief Parameters for the compression and decompression
 *
 * @par References
 *      - @ref CompressionLevel - Compression levels
 */
public class CompressionParameter extends Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
    * CompressionParameter constructor
    */
    public CompressionParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Sets the compression level
     * @param level   Compression level
     */
    public void setCompressionLevel(CompressionLevel level) {
        cSetCompressionLevel(this.cObject, level.getValue());
    }

    /**
     * Returns the compression level
     * @return Compression level
     */
    public CompressionLevel getCompressionLevel() {
        CompressionLevel cLevel = new CompressionLevel(cGetCompressionLevel(this.cObject));
        return cLevel;
    }

    private native void cSetCompressionLevel(long parAddr, int cLevel);

    private native int cGetCompressionLevel(long parAddr);
}
