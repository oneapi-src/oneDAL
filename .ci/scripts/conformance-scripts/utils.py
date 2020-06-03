# -*- coding: utf-8 -*-
from datetime import datetime

daalLine = "uses Intel® DAAL solver"
sklearnLine = "uses original Scikit-learn solver"
failLine = "uses original Scikit-learn solver, because the task was not solved with Intel® DAAL"

def make_report(algs_filename, report_filename):
    countDaalCalls = 0
    countSklearnCalls = 0
    countDaalFailCalls = 0

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

    for alg_name in algs:
        report_file.write("<br><h2>Testing %s</h2>" % alg_name)

        log_filename = "_log_%s.txt" % alg_name
        log_file = open(log_filename, "r")
        lines = log_file.readlines()
        log_file.close()

        countDaalCallsLocal = 0
        countSklearnCallsLocal = 0
        countDaalFailCallsLocal = 0
        result_str = ""
        reportAlgText = ""

        for line in lines:
            if daalLine in line:
                countDaalCallsLocal += 1
            if sklearnLine in line:
                countSklearnCallsLocal += 1
            if failLine in line:
                countDaalFailCallsLocal += 1
        

        for line in reversed(lines):
            if '=====' in line:
                result_str = line
                break
        
        countDaalCalls += countDaalCallsLocal
        countSklearnCalls += countSklearnCallsLocal
        countDaalFailCalls += countDaalFailCallsLocal

        countAllCallsLocal = countSklearnCallsLocal + countDaalCallsLocal
        percentDaalCallsLocal = float(countDaalCallsLocal - countDaalFailCallsLocal) / (countAllCallsLocal) * 100 if countAllCallsLocal else 0

        reportAlgText += "Number of Scikit-learn calls: %d <br>" % countSklearnCallsLocal
        reportAlgText += "Number of daal4py calls: %d <br>" % countDaalCallsLocal
        reportAlgText += "Number of daal4py fail calls: %d <br>" % countDaalFailCallsLocal
        reportAlgText += "Percent of using daal4py: %d %% <br>" % int(percentDaalCallsLocal)
        reportAlgText += line + "<br>"

        report_file.write(reportAlgText)

    report_file.write("<br><h1>Summary</h1>")

    countAllCalls = countSklearnCalls + countDaalCalls
    percentDaalCalls = float(countDaalCalls - countDaalFailCalls) / (countAllCalls) * 100 if countAllCalls else 0

    summaryText = "Number of Scikit-learn calls: %d <br>" % countSklearnCalls
    summaryText += "Number of daal4py calls: %d <br>" % countDaalCalls
    summaryText += "Number of daal4py fail calls: %d <br>" % countDaalFailCalls
    summaryText += "Percent of using daal4py: %d %% <br>" % int(percentDaalCalls)
    report_file.write(summaryText)

    textHTML = """<br>
    Finishing testing in """+str(datetime.now())+"""<br></p>
        </body>
    </html>"""

    report_file.write(textHTML)
    report_file.close()
