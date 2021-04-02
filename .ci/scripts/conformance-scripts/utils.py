#===============================================================================
# Copyright 2021 Intel Corporation
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

# -*- coding: utf-8 -*-
from datetime import datetime

class CallCounter:
    def __init__(self):
        self.clear()

    def clear(self):
        self.dalCalls = 0
        self.sklearnCalls = 0
        self.dalFailCalls = 0
        self.dalDeviceCallsRatio = 0.0
        self.dalDevicePatchedCalls = 0

        self.dalDeviceOffloadSuccess = 0
        self.dalDeviceOffloadFail = 0

    def calcDeviceCalls(self):
        offloadSum = self.dalDeviceOffloadSuccess + self.dalDeviceOffloadFail
        if offloadSum > 0:
            self.dalDeviceCallsRatio += self.dalDeviceOffloadSuccess / offloadSum
        self.dalDeviceOffloadSuccess = 0
        self.dalDeviceOffloadFail = 0

    def inc(self, other):
        self.dalCalls += other.dalCalls
        self.sklearnCalls += other.sklearnCalls
        self.dalFailCalls += other.dalFailCalls
        self.dalDeviceCallsRatio += other.dalDeviceCallsRatio
        self.dalDevicePatchedCalls += other.dalDevicePatchedCalls
        self.dalDeviceOffloadSuccess += other.dalDeviceOffloadSuccess
        self.dalDeviceOffloadFail += other.dalDeviceOffloadFail

class LineParser:

    def __init__(self, device=None, consider_fails=False):
        self.dalLine = "running accelerated version"
        self.sklearnLine = "fallback to original Scikit-learn"
        self.dalFailLine = "failed to run accelerated version, fallback to original Scikit-learn"

        if device != 'CPU':
            self.dalDeviceOffloadSuccessLine = f"successfully run on {device.lower()}"
            self.dalDeviceOffloadFailLine = f"failed to run on {device.lower()}. Fallback to host"
        self.device = device

        self.dalDeviceLine = f"{self.dalLine} on {self.device}"

        self.consider_fails = consider_fails

        self.algoCalls = CallCounter()
        self._localTestCalls = CallCounter()

    def clearCounters(self):
        self.algoCalls.clear()
        self._localTestCalls.clear()

    def parseLine(self, line):
        test_signal_fail = "FAILED"
        test_signal_pass = "PASSED"

        if self.dalLine in line:
            self._localTestCalls.dalCalls += 1
        if self.sklearnLine in line:
            self._localTestCalls.sklearnCalls += 1
        if self.dalFailLine in line:
            self._localTestCalls.dalFailCalls += 1
        if self.dalDeviceLine in line:
            self._localTestCalls.dalDevicePatchedCalls += 1

        if self.device != 'CPU':
            if self.dalDeviceOffloadSuccessLine in line:
                self._localTestCalls.dalDeviceOffloadSuccess += 1
            elif self.dalDeviceOffloadFailLine in line:
                self._localTestCalls.dalDeviceOffloadFail += 1
            else:
                self._localTestCalls.calcDeviceCalls()

        if test_signal_fail in line or test_signal_pass in line:
            self._localTestCalls.calcDeviceCalls()
            if not self.consider_fails or test_signal_pass in line:
                self.algoCalls.inc(self._localTestCalls)
            self._localTestCalls.clear()

def make_summory(counter, device):
    countAllCalls = counter.sklearnCalls + counter.dalCalls
    percentDalCalls = float(counter.dalCalls - counter.dalFailCalls) / (countAllCalls) * 100 if countAllCalls else 0

    # to calculate deviceOffloadCalls, we try to use dalDeivceCallsRatio as more fine-grained metric
    # if it is unavailable, we use dalDevicePatchedCalls count instead
    deviceOffloadCalls = counter.dalDeviceCallsRatio if counter.dalDeviceCallsRatio > 0 else counter.dalDevicePatchedCalls
    daal4pyOffloadPersent = float(deviceOffloadCalls / counter.dalCalls) * 100 if counter.dalCalls else 0
    totalOffloatPersent = percentDalCalls * daal4pyOffloadPersent / 100

    reportText = ""
    reportText += "Number of Scikit-learn calls: %d <br>" % counter.sklearnCalls
    reportText += "Number of daal4py calls: %d <br>" % counter.dalCalls
    reportText += "Number of daal4py fail calls: %d <br>" % counter.dalFailCalls
    reportText += "Percent of using daal4py: %d %% <br>" % int(percentDalCalls)
    if device != 'CPU':
        reportText += "Percent of daal4py calls offloaded to %s: %d %% <br>" % (device, int(daal4pyOffloadPersent))
        reportText += "Percent of using daal4py on %s: %d %% <br>" % (device, int(totalOffloatPersent))

    print('Number of Scikit-learn calls: %d' % counter.sklearnCalls)
    print('Number of daal4py calls: %d' % counter.dalCalls)
    print('Number of daal4py fail calls: %d' % counter.dalFailCalls)
    print('Percent of using daal4py: %d %%' % int(percentDalCalls))
    if device != 'CPU':
        print("Percent of daal4py calls offloaded to %s: %d %%" % (device, int(daal4pyOffloadPersent)))
        print("Percent of using daal4py on %s: %d %%" % (device, int(totalOffloatPersent)))

    return reportText


def device_from_sycl_terminology(device):
    if device == 'cpu' or device == 'host' or device is None:
        return 'CPU'
    if device == 'gpu':
        return 'GPU'
    else:
        raise ValueError(f"Unexpected device name {device}."
                         " Supported types are host, cpu and gpu")


def make_report(algs_filename, report_filename, device=None, consider_fails=False):
    device = device_from_sycl_terminology(device)

    with open(algs_filename, "r") as file_algs:
        algs = file_algs.read().split("\n")
    algs.remove("")

    report_file = open(report_filename,'wt')
    textHTML = """<html>
        <head>
        <title>
            Report of conformance testing
        </title>
        </head>
        <body>
            <p>
            Start of testing in """ + str(datetime.now())+ "<br>"
    report_file.write(textHTML)

    globalCalls = CallCounter()
    parser = LineParser(device, consider_fails)

    for alg_name in algs:
        parser.clearCounters()
        report_file.write("<br><h2>Testing %s</h2>" % alg_name)

        log_filename = "_log_%s.txt" % alg_name
        log_file = open(log_filename, "r")
        lines = log_file.readlines()
        log_file.close()

        result_str = ""

        for line in lines:
            if '====' in line and not 'test session starts' in line:
                break
            parser.parseLine(line)

        for line in reversed(lines):
            if '=====' in line:
                if 'test session starts' in line:
                    raise Exception('Found an error while testing %s' % (alg_name))
                result_str = line
                break

        globalCalls.inc(parser.algoCalls)

        if parser.algoCalls.dalCalls == 0 and parser.algoCalls.sklearnCalls == 0 and parser.algoCalls.dalFailCalls == 0:
            raise Exception('Algorithm %s has never been called' % (alg_name))

        print('*********************************************')
        print('Algorithm: %s' % alg_name)
        reportAlgText = make_summory(parser.algoCalls, device)
        report_file.write(reportAlgText)
        print(result_str)
        report_file.write(result_str + "<br>")

    print('*********************************************')
    print('Summary')
    report_file.write("<br><h1>Summary</h1>")
    summaryText = make_summory(globalCalls, device)
    report_file.write(summaryText)

    textHTML = """<br>
    Finishing testing in """+str(datetime.now())+"""<br></p>
        </body>
    </html>"""

    report_file.write(textHTML)
    report_file.close()
