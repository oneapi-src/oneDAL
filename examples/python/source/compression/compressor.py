# file: compressor.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

#
# !  Content:
# !    Python example of a compressor
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-COMPRESSOR"></a>
## \example compressor.py
#

import os
import sys

if sys.version[0] == '2':
    import Queue as Queue
else:
    import queue as Queue

import numpy as np

from daal.data_management import Compressor_Zlib, Decompressor_Zlib

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import getCRC32, readTextFile

datasetFileName = os.path.join('..', 'data', 'batch', 'logitboost_train.csv')

# Queue for sending and receiving compressed data blocks
sendReceiveQueue = Queue.Queue()

maxDataBlockSize = 16384  # Maximum size of a data block

def getUncompressedDataBlock(sentDataStream, availableDataSize):
    cur_pos = sentDataStream.size - availableDataSize

    # return next slice of array and remaining datasize
    if availableDataSize >= maxDataBlockSize:
        return (sentDataStream[cur_pos:cur_pos + maxDataBlockSize], availableDataSize - maxDataBlockSize)
    elif availableDataSize < maxDataBlockSize and availableDataSize > 0:
        return (sentDataStream[cur_pos:cur_pos + availableDataSize], 0)
    return (None,None)


def sendCompressedDataBlock(block):
    currentBlock = np.copy(block)
    # Push the current compressed block to the queue
    sendReceiveQueue.put(currentBlock)


def receiveCompressedDataBlock():
    # Stop at the end of the queue
    if sendReceiveQueue.empty():
        return None
    # Receive and copy the current compressed block from the queue
    return np.copy(sendReceiveQueue.get())


def printCRC32(sentDataStream, receivedDataStream):
    # Compute checksums for full input data and full received data
    crcSentDataStream = getCRC32(sentDataStream)
    crcReceivedDataStream = getCRC32(receivedDataStream)

    print("\nCompression example program results:\n")

    print("Input data checksum:    0x{:02X}".format(crcSentDataStream))
    print("Received data checksum: 0x{:02X}".format(crcReceivedDataStream))

    if sentDataStream.size != receivedDataStream.size:
        print("ERROR: Received data size mismatches with the sent data size")

    elif crcSentDataStream != crcReceivedDataStream:
        print("ERROR: Received data CRC mismatches with the sent data CRC")
    else:
        print("OK: Received data CRC matches with the sent data CRC")


if __name__ == "__main__":

    # Read data from a file
    sentDataStream = readTextFile(datasetFileName)

    # Allocate buffers for compressed and received data
    compressedDataBlock = np.empty(maxDataBlockSize, dtype=np.uint8)
    receivedDataStream = np.empty(sentDataStream.size, dtype=np.uint8)

    # Create a compressor
    compressor = Compressor_Zlib()

    # Receive the next data block for compression
    (uncompressedDataBlock, availableDataSize) = getUncompressedDataBlock(sentDataStream, sentDataStream.size)
    while uncompressedDataBlock is not None:
        # Associate data to compress with the compressor
        compressor.setInputDataBlock(uncompressedDataBlock, 0)

        # Memory for a compressed block might not be enough to compress the input block at once
        while True:
            # Compress uncompressedDataBlock to compressedDataBlock
            compressor.run(compressedDataBlock, 0)

            # Get the actual size of a compressed block
            compressedDataView = compressedDataBlock[0:compressor.getUsedOutputDataBlockSize()]

            # Send the current compressed block
            sendCompressedDataBlock(compressedDataView)

            # Check if an additional data block is needed to complete compression
            if not compressor.isOutputDataBlockFull():
                break

        # Receive the next data block for compression
        (uncompressedDataBlock, availableDataSize) = getUncompressedDataBlock(sentDataStream, availableDataSize)

    # Create a decompressor
    decompressor = Decompressor_Zlib()

    # Receive compressed data by blocks
    compressedDataBlock = receiveCompressedDataBlock()
    offset = 0

    while compressedDataBlock is not None:
        # Associate compressed data with the decompressor
        decompressor.setInputDataBlock(compressedDataBlock, 0)

        # Decompress an incoming block to the end of receivedDataStream
        decompressor.run(receivedDataStream, offset)

        # Update the size of actual data in receivedDataStream
        offset += decompressor.getUsedOutputDataBlockSize()

        # Receive next block
        compressedDataBlock = receiveCompressedDataBlock()

    # Compute and print checksums for sentDataStream and receivedDataStream
    printCRC32(sentDataStream, receivedDataStream)
