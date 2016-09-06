/* file: Compressor.java */
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

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__COMPRESSOR"></a>
 * @brief The base class that provides methods for the compression
 * \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a>
 *
 * @par References
 *      - @ref CompressionMethod class
 */
public class Compressor extends Compression {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Compressor constructor
     * @param context   Context to manage created compressor
     * @param method Compression method, @ref CompressionMethod
     */
    public Compressor(DaalContext context, CompressionMethod method) {
        super(context);
        if (method != CompressionMethod.zlib && method != CompressionMethod.lzo && method != CompressionMethod.rle
                && method != CompressionMethod.bzip2) {
            throw new IllegalArgumentException("method unsupported");
        }
        this.cObject = cInit(method.getValue());
    }

    private native long cInit(int comprMethod);

    protected native long cInitParameter(long comprAddr, int comprMethod);
}
