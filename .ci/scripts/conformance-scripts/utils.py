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
        self.sklearnexCalls = 0
        self.sklearnCalls = 0
        self.sklearnexFailCalls = 0
        self.sklearnexDeviceCallsRatio = 0.0
        self.sklearnexDevicePatchedCalls = 0

        self.sklearnexDeviceOffloadSuccess = 0
        self.sklearnexDeviceOffloadFail = 0

    def calcDeviceCalls(self):
        offloadSum = self.sklearnexDeviceOffloadSuccess + self.sklearnexDeviceOffloadFail
        if offloadSum > 0:
            self.sklearnexDeviceCallsRatio += self.sklearnexDeviceOffloadSuccess / offloadSum
        self.sklearnexDeviceOffloadSuccess = 0
        self.sklearnexDeviceOffloadFail = 0

    def inc(self, other):
        self.sklearnexCalls += other.sklearnexCalls
        self.sklearnCalls += other.sklearnCalls
        self.sklearnexFailCalls += other.sklearnexFailCalls
        self.sklearnexDeviceCallsRatio += other.sklearnexDeviceCallsRatio
        self.sklearnexDevicePatchedCalls += other.sklearnexDevicePatchedCalls
        self.sklearnexDeviceOffloadSuccess += other.sklearnexDeviceOffloadSuccess
        self.sklearnexDeviceOffloadFail += other.sklearnexDeviceOffloadFail

class LineParser:

    def __init__(self, device=None, consider_fails=False):
        self.sklearnexLine = "running accelerated version"
        self.sklearnLine = "fallback to original Scikit-learn"
        self.sklearnexFailLine = "failed to run accelerated version, fallback to original Scikit-learn"

        if device != 'CPU':
            self.sklearnexDeviceOffloadSuccessLine = f"successfully run on {device.lower()}"
            self.sklearnexDeviceOffloadFailLine = f"failed to run on {device.lower()}. Fallback to host"
        self.device = device

        self.sklearnexDeviceLine = f"{self.sklearnexLine} on {self.device}"

        self.consider_fails = consider_fails

        self.algoCalls = CallCounter()
        self._localTestCalls = CallCounter()

    def clearCounters(self):
        self.algoCalls.clear()
        self._localTestCalls.clear()

    def parseLine(self, line):
        test_signal_fail = "FAILED"
        test_signal_pass = "PASSED"

        if self.sklearnexLine in line:
            self._localTestCalls.sklearnexCalls += 1
        if self.sklearnLine in line:
            self._localTestCalls.sklearnCalls += 1
        if self.sklearnexFailLine in line:
            self._localTestCalls.sklearnexFailCalls += 1
        if self.sklearnexDeviceLine in line:
            self._localTestCalls.sklearnexDevicePatchedCalls += 1

        if self.device != 'CPU':
            if self.sklearnexDeviceOffloadSuccessLine in line:
                self._localTestCalls.sklearnexDeviceOffloadSuccess += 1
            elif self.sklearnexDeviceOffloadFailLine in line:
                self._localTestCalls.sklearnexDeviceOffloadFail += 1
            else:
                self._localTestCalls.calcDeviceCalls()

        if test_signal_fail in line or test_signal_pass in line:
            self._localTestCalls.calcDeviceCalls()
            if not self.consider_fails or test_signal_pass in line:
                self.algoCalls.inc(self._localTestCalls)
            self._localTestCalls.clear()

def make_summory(counter, device):
    countAllCalls = counter.sklearnCalls + counter.sklearnexCalls
    percentDalCalls = float(counter.sklearnexCalls - counter.sklearnexFailCalls) / (countAllCalls) * 100 if countAllCalls else 0

    # to calculate deviceOffloadCalls, we try to use sklearnexDeivceCallsRatio as more fine-grained metric
    # if it is unavailable, we use sklearnexDevicePatchedCalls count instead
    deviceOffloadCalls = counter.sklearnexDeviceCallsRatio if counter.sklearnexDeviceCallsRatio > 0 else counter.sklearnexDevicePatchedCalls
    sklearnexOffloadPersent = float(deviceOffloadCalls / counter.sklearnexCalls) * 100 if counter.sklearnexCalls else 0
    totalOffloatPersent = percentDalCalls * sklearnexOffloadPersent / 100

    reportText = ""
    reportText += "Number of Scikit-learn calls: %d <br>" % counter.sklearnCalls
    reportText += "Number of sklearnex calls: %d <br>" % counter.sklearnexCalls
    reportText += "Number of sklearnex fail calls: %d <br>" % counter.sklearnexFailCalls
    reportText += "Percent of using sklearnex: %d %% <br>" % int(percentDalCalls)
    if device != 'CPU':
        reportText += "Percent of sklearnex calls offloaded to %s: %d %% <br>" % (device, int(sklearnexOffloadPersent))
        reportText += "Percent of using sklearnex on %s: %d %% <br>" % (device, int(totalOffloatPersent))

    print('Number of Scikit-learn calls: %d' % counter.sklearnCalls)
    print('Number of sklearnex calls: %d' % counter.sklearnexCalls)
    print('Number of sklearnex fail calls: %d' % counter.sklearnexFailCalls)
    print('Percent of using sklearnex: %d %%' % int(percentDalCalls))
    if device != 'CPU':
        print("Percent of sklearnex calls offloaded to %s: %d %%" % (device, int(sklearnexOffloadPersent)))
        print("Percent of using sklearnex on %s: %d %%" % (device, int(totalOffloatPersent)))

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

        if parser.algoCalls.sklearnexCalls == 0 and parser.algoCalls.sklearnCalls == 0 and parser.algoCalls.sklearnexFailCalls == 0:
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
