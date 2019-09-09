# file: compression_batch.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

#
# !  Content:
# !    Python example of compression in the batch processing mode
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-COMPRESSION_BATCH"></a>
## \example compression_batch.py
#

import os
import sys

import numpy as np

from daal.data_management import Compressor_Zlib, Decompressor_Zlib, level9, DecompressionStream, CompressionStream

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import getCRC32, readTextFile

DATA_PREFIX = os.path.join('..', 'data', 'batch')
datasetFileName = os.path.join(DATA_PREFIX, 'logitboost_train.csv')


def printCRC32(rawData, deCompressedData):

    # Compute checksums for raw data and the decompressed data
    crcRawData = getCRC32(rawData)
    crcDecompressedData = getCRC32(deCompressedData)

    print("\nCompression example program results:\n")

    print("Raw data checksum:    0x{:02X}".format(crcRawData))
    print("Decompressed data checksum: 0x{:02X}".format(crcDecompressedData))

    if rawData.size != deCompressedData.size:
        print("ERROR: Decompressed data size mismatches with the raw data size")

    elif crcRawData != crcDecompressedData:
        print("ERROR: Decompressed data CRC mismatches with the raw data CRC")

    else:
        print("OK: Decompressed data CRC matches with the raw data CRC")


if __name__ == "__main__":
    # Read data from a file
    rawData = readTextFile(datasetFileName)

    # Create a compressor
    compressor = Compressor_Zlib()
    compressor.parameter.gzHeader = True
    compressor.parameter.level = level9

    # Create a stream for compression
    comprStream = CompressionStream(compressor)

    # Write raw data to the compression stream and compress if needed
    comprStream.push_back(rawData)

    # Allocate memory to store the compressed data
    compressedData = np.empty(comprStream.getCompressedDataSize(), dtype=np.uint8)

    # Store the compressed data
    comprStream.copyCompressedArray(compressedData)

    # Create a decompressor
    decompressor = Decompressor_Zlib()
    decompressor.parameter.gzHeader = True

    # Create a stream for decompression
    deComprStream = DecompressionStream(decompressor)

    # Write the compressed data to the decompression stream and decompress it
    deComprStream.push_back(compressedData)

    # Allocate memory to store the decompressed data
    deCompressedData = np.empty(deComprStream.getDecompressedDataSize(), dtype=np.uint8)

    # Store the decompressed data
    deComprStream.copyDecompressedArray(deCompressedData)

    # Compute and print checksums for raw data and the decompressed data
    printCRC32(rawData, deCompressedData)
