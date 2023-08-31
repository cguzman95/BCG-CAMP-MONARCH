#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

import matplotlib as mpl

mpl.use('TkAgg')
#import plot_functions #comment to save ~2s execution time
import math_functions
import sys, getopt
import os
import numpy as np
import datetime
import json
from pathlib import Path
import zipfile
from os import walk
import subprocess


class cmakeVarsClass:
    def __init__(self):
        self.caseImpl = ""
        self.maxrregcount = ""

class TestMonarch:
    def __init__(self):
        # Case configuration
        #self.confCase = confCaseClass()
        self.cmakeVars = cmakeVarsClass()
        self._chemFile = "monarch_binned"
        self.diffCells = ""
        self.mpi = "yes"
        self.timeSteps = 1
        self.timeStepsDt = 2
        self.MAPETol = 1.0E-4
        self.commit = ""
        self.case = []
        self.nCells = 1
        self.caseGpuCpu = ""
        self.caseImpl = ""
        self.mpiProcesses = 1
        self.allocatedTasksPerNode = 160
        self.nGPUs = 1
        # Cases configuration
        self.is_start_cases_attributes = True
        self.cmakeVarsBase = cmakeVarsClass()
        self.cmakeVarsOptim = cmakeVarsClass()
        self.diffCellsL = ["Realistic"]
        self.mpiProcessesCaseBase = 1
        self.mpiProcessesCaseOptimList = []
        self.nGPUsCaseOptimList = [1]
        self.cells = [100]
        self.caseBase = ""
        self.casesOptim = [""]
        self.plotYKey = ""
        self.plotXKey = ""
        self.is_export = False
        self.is_import = False
        self.profileCuda = ""
        # Auxiliary
        self.is_start_auxiliary_attributes = True
        self.is_make = True
        self.sbatch_job_id = ""
        self.datacolumns = []
        self.stdColumns = []
        self.exportPath = "test/monarch/exports"
        self.legend = []
        self.results_file = "_solver_stats.csv"
        self.plotTitle = ""
        self.nCellsProcesses = 1
        self.itsolverConfigFile = "itsolver_options.txt"
        self.campSolverConfigFile = "config_variables_c_solver.txt"

    @property
    def chemFile(self):
        return self._chemFile

    @chemFile.setter
    def chemFile(self, new_chemFile):
        self._chemFile = new_chemFile


def getCaseName(conf):
    case_multicells_onecell_name = ""
    # if conf.caseImpl != "BDF" and conf.caseGpuCpu == "GPU":
    # case_multicells_onecell_name = "LS "
    if conf.caseImpl == "Block-cellsN":
        case_multicells_onecell_name += "Block-cells (N)"
    elif conf.caseImpl == "Block-cells1":
        case_multicells_onecell_name += "Block-cells (1)"
    elif conf.caseImpl == "Block-cellsNhalf":
        case_multicells_onecell_name += "Block-cells (N/2)"
    elif conf.caseImpl.find("maxrregcount") != -1:
        case_multicells_onecell_name += ""
        print("WARNING: Changed name maxrregcount to", case_multicells_onecell_name)
        # case_multicells_onecell_name += conf.caseImpl
    elif conf.caseImpl.find("One") != -1:
        case_multicells_onecell_name += "Base version"
    else:
        case_multicells_onecell_name += conf.caseImpl

    return case_multicells_onecell_name

def writeConfBCG(conf):
    file1 = open("../data/conf.txt", "w")

    file1.write(conf.chemFile+"\n")
    file1.write(str(conf.nGPUs)+"\n")
    file1.write(str(conf.nCells)+"\n")
    #file1.write(str(conf.timeSteps)+"\n")

    file1.close()

    conf_path = "../data/confCmakeVars.json"
    with open(conf_path, 'w', encoding='utf-8') as jsonFile:
        json.dump(conf.cmakeVars.__dict__, jsonFile, indent=4, sort_keys=False)


def run(conf):
    exec_str = ""
    if conf.mpi == "yes":
        exec_str += "mpirun -v -np " + str(conf.mpiProcesses) + " --bind-to core "
        # exec_str+="srun -n "+str(conf.mpiProcesses)+" "

    if conf.profileCuda=="nvprof" and conf.caseGpuCpu == "GPU":
        pathNvprof = "../nvprof/"
        Path(pathNvprof).mkdir(parents=True, exist_ok=True)
        pathNvprof = "../nvprof/" + conf.caseImpl \
                     + str(conf.nCells) + "Cells" +  ".nvprof "
        #exec_str += "nvprof --analysis-metrics -f -o " + pathNvprof #all metrics
        exec_str += "nvprof --print-gpu-trace " #registers per thread
        # --print-gpu-summary
        print("Saving Nvprof file in ", os.path.abspath(os.getcwd()) \
              + "/" + pathNvprof)
    elif conf.profileCuda=="nsight" and conf.caseGpuCpu == "GPU":
        pathNvprof = "../nsight/"
        Path(pathNvprof).mkdir(parents=True, exist_ok=True)
        pathNvprof = "../nsight/" + conf.caseImpl \
                     + str(conf.nCells) + "Cells "
        #exec_str += "/apps/NVIDIA-HPC-SDK/21.3/Linux_ppc64le/21.3/profilers/Nsight_Compute/ncu --set full -f -o " + pathNvprof
        exec_str += "/apps/NVIDIA-HPC-SDK/21.3/Linux_ppc64le/21.3/profilers/Nsight_Compute/ncu "

    # --print-gpu-summary
        print("Saving nsight file in ", os.path.abspath(os.getcwd()) \
          + "/" + pathNvprof)

    exec_str += "./test"

    data_name = "timesAndCounters.csv"# conf.chemFile + '_' + conf.caseImpl + conf.results_file
    tmp_path = 'out/' + data_name

    if conf.is_import and conf.plotYKey != "MAPE":
        is_import, data_path = import_data(conf, tmp_path)
    else:
        is_import, data_path = False, tmp_path

    if not is_import:
        #with open("../data/conf.txt", 'r') as fp:
            #caseImplImported = fp.readlines()[3].strip()
        #print(caseImplImported, conf.caseImpl)

        conf_name="../data/confCmakeVars.json"
        jsonFile = open(conf_name)
        conf_imported = json.load(jsonFile)
        conf_dict = vars(conf.cmakeVars)
        print("conf_dict",conf_dict)
        #print("conf_imported",conf_imported)

        cmake_str = "cmake ."
        for confKey in conf_dict:
            if conf_imported[confKey] != conf_dict[confKey]:
                if conf_imported[confKey] != "":
                    cmake_str += " -D" + conf_imported[confKey] + "=OFF"
                if conf_dict[confKey] != "":
                    cmake_str += " -D" + conf_dict[confKey] + "=ON"
                conf.is_make = True
                os.system(cmake_str)
                #print(cmake_str)

        if conf.is_make:
            os.system("make -j 4")
            conf.is_make = False

        #if caseImplImported != conf.caseImpl:
         #   cmake_str = "cmake . -D" + caseImplImported + "=OFF" + \
           #           " -D" + conf.caseImpl + "=ON"
            #os.system(cmake_str)
            #os.system("make -j 4")

            #conf.is_make = True

        #if conf.is_make:
        #    os.system("make -j 4")
         #   conf.is_make = False

        writeConfBCG(conf)

        print("exec_str:", exec_str, "ncells", conf.nCells)#, conf.diffCells, conf.caseGpuCpu, conf.caseImpl, conf.mpiProcesses,conf.nGPUs)
        os.system(exec_str)
        if conf.is_export:
            export(conf, data_path)

    data = {}
    if conf.plotYKey == "MAPE":
        # print("Pending to import data from MAPE and read only the desired timesteps and cells")
        plot_functions.read_solver_stats_all(data_path, data)
    else:
        plot_functions.read_solver_stats(data_path, data, conf.timeSteps)

    # print("The size of the dictionary is {} bytes".format(sys.getsizeof(data)))
    # print("The size of the dictionary is {} bytes".format(sys.getsizeof(data["timeLS"])))

    if is_import:
        os.remove(data_path)

    return data


def run_case(conf):
    data = run(conf)

    if "timeLS" in conf.plotYKey and "computational" in conf.plotYKey \
            and "GPU" in conf.case:
        for i in range(len(data["timeLS"])):
            data["timeLS"][i] = data["timeLS"][i] - data["timeBiconjGradMemcpy"][i]

    y_key_words = conf.plotYKey.split()
    y_key = y_key_words[-1]
    if "normalized" in conf.plotYKey:
        nSystemsOfCells = 1
        if "One-cell" in conf.case:
            nSystemsOfCells = conf.nCells
        if y_key == "timeLS":
            for i in range(len(data[y_key])):
                data[y_key][i] = data[y_key][i] / (data["counterLS"][i] / nSystemsOfCells)
        elif y_key == "timecvStep":
            for i in range(len(data[y_key])):
                data[y_key][i] = data[y_key][i] / (data["countercvStep"][i] * nSystemsOfCells)
        else:  # counterBCG and other counters
            for i in range(len(data[y_key])):
                data[y_key][i] = data[y_key][i] / nSystemsOfCells

    if "(Comp.timeLS/counterBCG)" in conf.plotYKey and "GPU" in conf.case:
        for i in range(len(data["timeLS"])):
            data["timeLS"][i] = data["timeLS"][i] - data["timeBiconjGradMemcpy"][i]
        for i in range(len(data["timeLS"])):
            data["timeLS"][i] = data["timeLS"][i] \
                                / data["counterBCG"][i]

        for j in range(len(data["timeLS"])):
            data["timeLS"][j] = data["timeLS"][j] \
                                / data["counterBCG"][j]

    # if conf.plotYKey != "MAPE":
    #    print("run_case", conf.case, y_key, ":", data[y_key])

    return data


def run_cases(conf):
    # Run base case
    conf.mpiProcesses = conf.mpiProcessesCaseBase
    conf.nCells = int(conf.nCellsProcesses / conf.mpiProcesses)
    conf.nGPUs = conf.nGPUsCaseBase

    cases_words = conf.caseBase.split()
    conf.caseGpuCpu = cases_words[0]
    conf.caseImpl = cases_words[1]
    conf.cmakeVarsBase.caseImpl = cases_words[1]
    conf.cmakeVars = conf.cmakeVarsBase

    conf.case = conf.caseBase
    dataCaseBase = run_case(conf)
    data = {"caseBase": dataCaseBase}

    # Run OptimCases
    datacases = []
    stdCases = []
    for nGPUs in conf.nGPUsCaseOptimList:
        conf.nGPUs = nGPUs
        for mpiProcessesCaseOptim in conf.mpiProcessesCaseOptimList:
            conf.mpiProcesses = mpiProcessesCaseOptim
            conf.nCells = int(conf.nCellsProcesses / conf.mpiProcesses)
            for caseOptim in conf.casesOptim:
                if conf.plotXKey == "MPI processes":
                    if (caseOptim == conf.caseBase and mpiProcessesCaseOptim == conf.mpiProcessesCaseBase) \
                            or (caseOptim != conf.caseBase and mpiProcessesCaseOptim != conf.mpiProcessesCaseBase):
                        continue

                cases_words = caseOptim.split()
                conf.caseGpuCpu = cases_words[0]
                conf.caseImpl = cases_words[1]
                conf.cmakeVarsOptim.caseImpl = cases_words[1]
                conf.cmakeVars = conf.cmakeVarsOptim

                conf.case = caseOptim
                data["caseOptim"] = run_case(conf)

                # calculate measures between caseBase and caseOptim
                if conf.plotYKey == "NRMSE":
                    datay = plot_functions.calculate_NMRSE(data, conf.timeSteps)
                elif conf.plotYKey == "MAPE":
                    datay = plot_functions.calculate_MAPE(data, conf.timeSteps, conf.MAPETol)
                elif conf.plotYKey == "SMAPE":
                    datay = plot_functions.calculate_SMAPE(data, conf.timeSteps)
                elif "Speedup" in conf.plotYKey:
                    y_key_words = conf.plotYKey.split()
                    y_key = y_key_words[-1]
                    # print("WARNING: Check y_key is correct:",y_key)
                    datay = plot_functions.calculate_speedup(data, y_key)
                elif conf.plotYKey == "Percentage data transfers CPU-GPU [%]":
                    y_key = "timeBiconjGradMemcpy"
                    print("elif conf.plotYKey==Time data transfers")
                    datay = plot_functions.calculate_BCGPercTimeDataTransfers(data, y_key)
                else:
                    raise Exception("Not found plot function for conf.plotYKey")

                if len(conf.cells) > 1 or conf.plotXKey == "MPI processes":
                    datacases.append(round(np.mean(datay), 2))
                    stdCases.append(round(np.std(datay), 2))
                    # print("datacases",datacases)
                    # print("stdCases",stdCases)
                else:
                    # datacases.append([round(elem, 2) for elem in datay])
                    datacases.append([round(elem, 2) for elem in datay])

    return datacases, stdCases


def run_cells(conf):
    datacells = []
    stdCells = []
    for i in range(len(conf.cells)):
        conf.nCellsProcesses = conf.cells[i]
        datacases, stdCases = run_cases(conf)

        # print("datacases",datacases)
        # print("stdCases",stdCases)

        if len(conf.cells) == 1:
            datacells = datacases
            stdCells = stdCases
        else:
            datacells.append(datacases)
            stdCells.append(stdCases)

    # print("datacells",datacells)

    if len(conf.cells) > 1:
        datacellsTranspose = np.transpose(datacells)
        datacells = datacellsTranspose.tolist()
        stdCellsTranspose = np.transpose(stdCells)
        stdCells = stdCellsTranspose.tolist()

    return datacells, stdCells


# Anything regarding different initial conditions is applied to both cases (Base and Optims/s)
def run_diffCells(conf):
    conf.datacolumns = []
    conf.stdColumns = []
    for i, diff_cells in enumerate(conf.diffCellsL):
        conf.diffCells = diff_cells
        datacells, stdcells = run_cells(conf)
        conf.datacolumns += datacells
        conf.stdColumns += stdcells


def plot_cases(conf):
    # Set plot info
    cases_words = conf.caseBase.split()
    conf.caseGpuCpu = cases_words[0]
    conf.caseImpl = cases_words[1]
    case_multicells_onecell_name = getCaseName(conf)
    # if conf.caseImpl.find("One-cell") != -1:
    #    case_multicells_onecell_name = "Base version"

    case_gpu_cpu_name = ""
    if conf.caseGpuCpu == "CPU":
        case_gpu_cpu_name = str(conf.mpiProcessesCaseBase) + " MPI" + " CPU"
    elif conf.caseGpuCpu == "GPU":
        if conf.mpiProcessesCaseBase > 1:
            case_gpu_cpu_name += str(conf.mpiProcessesCaseBase) + " MPI "
        case_gpu_cpu_name += str(conf.nGPUsCaseBase) + " GPU"
    else:
        case_gpu_cpu_name = conf.caseGpuCpu

    baseCaseName = ""
    if conf.plotYKey != "Percentage data transfers CPU-GPU [%]":  # Speedup
        baseCaseName = "vs " + case_gpu_cpu_name + " " + case_multicells_onecell_name

    conf.legend = []
    cases_words = conf.casesOptim[0].split()
    conf.caseGpuCpu = cases_words[0]
    conf.caseImpl = cases_words[1]
    last_arch_optim = conf.caseGpuCpu
    last_case_optim = conf.caseImpl
    is_same_arch_optim = True
    is_same_case_optim = True
    for caseOptim in conf.casesOptim:
        cases_words = caseOptim.split()
        conf.caseGpuCpu = cases_words[0]
        conf.caseImpl = cases_words[1]
        if last_arch_optim != conf.caseGpuCpu:
            is_same_arch_optim = False
        last_arch_optim = conf.caseGpuCpu
        # print(last_case_optim,conf.caseImpl)
        if last_case_optim != conf.caseImpl:
            is_same_case_optim = False
        last_case_optim = conf.caseImpl

    is_same_diff_cells = False
    for diff_cells in conf.diffCellsL:
        conf.diffCells = diff_cells
        for nGPUs in conf.nGPUsCaseOptimList:
            for mpiProcessesCaseOptim in conf.mpiProcessesCaseOptimList:
                for caseOptim in conf.casesOptim:
                    if conf.plotXKey == "MPI processes":
                        if (caseOptim == conf.caseBase and mpiProcessesCaseOptim == conf.mpiProcessesCaseBase) \
                                or (caseOptim != conf.caseBase and mpiProcessesCaseOptim != conf.mpiProcessesCaseBase):
                            continue
                    cases_words = caseOptim.split()
                    conf.caseGpuCpu = cases_words[0]
                    conf.caseImpl = cases_words[1]
                    case_multicells_onecell_name = getCaseName(conf)
                    if conf.caseImpl.find("BDF") != -1 or conf.caseImpl.find(
                            "maxrregcount") != -1:
                        is_same_diff_cells = True
                    legend_name = ""
                    if len(conf.diffCellsL) > 1:
                        legend_name += conf.diffCells + " "
                    if len(conf.mpiProcessesCaseOptimList) > 1 and conf.plotXKey == "MPI processes":
                        legend_name += str(mpiProcessesCaseOptim) + " MPI "
                    if len(conf.nGPUsCaseOptimList) > 1 and conf.caseGpuCpu == "GPU":
                        legend_name += str(nGPUs) + "GPU "
                    elif not is_same_arch_optim:
                        legend_name += conf.caseGpuCpu + " "
                    if not is_same_case_optim:
                        legend_name += case_multicells_onecell_name
                    if not legend_name == "":
                        conf.legend.append(legend_name)

    conf.plotTitle = ""
    #if not is_same_diff_cells and len(conf.diffCellsL) == 1:
    #    conf.plotTitle += conf.diffCells + " test: "
    if len(conf.mpiProcessesCaseOptimList) == 1 and conf.caseGpuCpu == "CPU":
        # if len(conf.mpiProcessesCaseOptimList) == 1:
        conf.plotTitle += str(mpiProcessesCaseOptim) + " MPI "
    if len(conf.nGPUsCaseOptimList) == 1 and conf.plotXKey == "GPUs":
        # conf.plotTitle += str(nGPUs) + " GPUs "
        conf.plotTitle += " GPUs "
    # print("is_same_arch_optim",is_same_arch_optim)
    if is_same_arch_optim:
        if conf.plotXKey == "MPI processes":
            conf.plotTitle += "CPU "
        elif conf.plotXKey == "GPUs":
            conf.plotTitle += ""
        else:
            # conf.plotTitle += str(nGPUs) + " "
            conf.plotTitle += conf.caseGpuCpu + " "
    if len(conf.legend) == 1 or not conf.legend or len(conf.diffCellsL) > 1:
        conf.plotTitle += case_multicells_onecell_name + " "
        if len(conf.diffCellsL) > 1:
            conf.plotTitle += "Implementations "
    else:
        if conf.plotXKey == "GPUs":
            conf.plotTitle += "GPU "
        if conf.plotXKey == "MPI processes":
            conf.plotTitle += "Speedup against 1 MPI CPU-based version"
        else:
            conf.plotTitle += "Implementations "
    if not conf.plotXKey == "MPI processes":
        conf.plotTitle += baseCaseName

    namey = conf.plotYKey
    if conf.plotYKey == "Speedup normalized computational timeLS":
        namey = "Speedup linear solver kernel"
    if conf.plotYKey == "Speedup counterLS":
        namey = "Speedup iterations CAMP solving"
    if conf.plotYKey == "Speedup normalized timeLS":
        namey = "Speedup linear solver"
    if conf.plotYKey == "Speedup timecvStep":
        namey = "Speedup"
    if conf.plotYKey == "Speedup countercvStep":
        namey = "Speedup iterations BDF loop"
    if conf.plotYKey == "Speedup timeCVode":
        namey = "Speedup CAMP solving"
    if conf.plotYKey == "MAPE":
        namey = "MAPE [%]"
    if conf.plotYKey == "Speedup counterBCG":
        namey = "Speedup solving iterations BCG"

    if len(conf.cells) > 1:
        namey += " [Mean and \u03C3]"
        # namey += " [Average]"
        print_timesteps_title = True
        #print_timesteps_title = False
        if print_timesteps_title:
            conf.plotTitle += ", " + str(conf.timeSteps)+" timesteps"
        datax = conf.cells
        plot_x_key = "Cells"
    elif conf.plotXKey == "MPI processes":
        conf.plotTitle += ", Cells: " + str(conf.cells[0])
        datax = conf.mpiProcessesCaseOptimList
        plot_x_key = conf.plotXKey
    elif conf.plotXKey == "GPUs":
        conf.plotTitle += ", Cells: " + str(conf.cells[0])
        datax = conf.nGPUsCaseOptimList
        plot_x_key = conf.plotXKey
    else:
        conf.plotTitle += ", Cells: " + str(conf.cells[0])
        datax = list(range(1, conf.timeSteps + 1, 1))
        plot_x_key = "Timesteps"

    namex = plot_x_key
    datay = conf.datacolumns

    if namex == "Timesteps":
        print("Mean:", round(np.mean(datay), 2))
        print("Std", round(np.std(datay), 2))
    else:
        print("Std", conf.stdColumns)

    if conf.cmakeVarsOptim.maxrregcount == "":
        conf.cmakeVarsOptim.maxrregcount = "maxrregcountAuto"
    if conf.cmakeVarsBase.maxrregcount == "":
        conf.cmakeVarsBase.maxrregcount = "maxrregcountAuto"
    print(conf.cmakeVarsOptim.maxrregcount, "vs", conf.cmakeVarsBase.maxrregcount)

    print(namex, ":", datax)
    print("plotTitle: ", conf.plotTitle, " legend:", conf.legend)
    print(namey, ":", datay)

    #plot_functions.plotsns(namex, namey, datax, datay, conf.stdColumns, conf.plotTitle, conf.legend)


def all_timesteps():
    conf = TestMonarch()

    conf.chemFile = "confBCG1Cell.txt"
    #conf.chemFile = "confBCG10Cells.txt"

    conf.profileCuda = ""
    #conf.profileCuda = "nvprof"
    #conf.profileCuda = "nsight"

    conf.nGPUsCaseBase = 1
    # conf.nGPUsCaseBase = 4

    conf.nGPUsCaseOptimList = [1]
    # conf.nGPUsCaseOptimList = [1]
    # conf.nGPUsCaseOptimList = [1,2,3,4]

    conf.mpi = "no"

    conf.mpiProcessesCaseBase = 1
    # conf.mpiProcessesCaseBase = 40

    conf.mpiProcessesCaseOptimList.append(1)
    #conf.mpiProcessesCaseOptimList.append(40)
    # conf.mpiProcessesCaseOptimList = [10,20,40]
    # conf.mpiProcessesCaseOptimList = [1,4,8,16,32,40]

    conf.allocatedTasksPerNode = 160
    # conf.allocatedTasksPerNode = 40
    # conf.allocatedTasksPerNode = 320
    # conf.allocatedTasksPerNode = get_ntasksPerNode_sbatch() #todo

    conf.cells = [100]
    #conf.cells = [100, 1000, 10000, 100000]

    conf.timeSteps = 1
    #conf.timeSteps = 720

    conf.timeStepsDt = 2

    conf.cmakeVarsBase.maxrregcount = ""
    #conf.cmakeVarsBase.maxrregcount = "use_maxrregcount32"

    #conf.cmakeVarsOptim.maxrregcount = ""
    #conf.cmakeVarsOptim.maxrregcount = "use_maxrregcount32"

    #conf.caseBase = "GPU CSC_ATOMIC"
    conf.caseBase = "GPU CSR"
    #conf.caseBase = "GPU CSR_1INDEX"
    #conf.caseBase = "GPU CUID"
    #conf.caseBase = "GPU CSD"
    #conf.caseBase = "GPU CSR_VECTOR" #Error
    #conf.caseBase = "GPU CSR_ADAPTIVE"
    #conf.caseBase = "GPU CSR_SHARED"
    #conf.caseBase = "GPU CSR_SHARED_DB"
    #conf.caseBase = "GPU CSR_SHARED_DB_JAC"

    conf.casesOptim = []
    #conf.casesOptim.append("GPU CSR")
    #conf.casesOptim.append("GPU CSC_ATOMIC")
    #conf.casesOptim.append("GPU CUID")
    #conf.casesOptim.append("GPU CSD")
    #conf.casesOptim.append("GPU CSR_SHARED")
    #conf.casesOptim.append("GPU CSR_SHARED_DB")
    #conf.casesOptim.append("GPU CSR_SHARED_DB_JAC")

    conf.plotYKey = "Speedup timeBiConjGrad"

    """END OF CONFIGURATION VARIABLES"""

    # Utility functions
    # remove_to_tmp(conf,"1653646090223272794")

    conf.results_file = "_solver_stats.csv"
    #os.chdir("../../..")

    if conf.plotYKey == "NRMSE" or conf.plotYKey == "MAPE" or conf.plotYKey == "SMAPE":
        conf.results_file = '_results_all_cells.csv'

    if not os.path.exists('out'):
        os.makedirs('out')

    if conf.chemFile == "monarch_binned":
        if conf.timeStepsDt != 2:
            print("Warning: Setting timeStepsDt to 2, since it is the usual value for monarch_binned")
        conf.timeStepsDt = 2
    elif conf.chemFile == "monarch_cb05":
        if conf.timeStepsDt != 3:
            print("Warning: Setting timeStepsDt to 3, since it is the usual value for monarch_cb05")
        conf.timeStepsDt = 3
        if "Realistic" in conf.diffCellsL:
            print("Warning: Setting Ideal, chemFile == monarch_cb05 only has information from one-cell and we do not know if is fine using different temp and pressure (e.g. converge or not)")
            #conf.diffCellsL = ["Realistic"]
            conf.diffCellsL = ["Ideal"]

    if not conf.caseBase:
        print("ERROR: caseBase is empty")
        raise

    if conf.caseBase  == "CPU EBI":
        print("Warning: Disable CAMP_PROFILING in CVODE to better profiling")
    if conf.caseBase == "CPU EBI" and conf.chemFile != "monarch_cb05":
        print("Error: Set conf.chemFile = monarch_cb05 to run CPU EBI")
        raise Exception
    for caseOptim in conf.casesOptim:
        if caseOptim == "CPU EBI":
            print("Warning: Disable CAMP_PROFILING in CVODE to better profiling")
        if caseOptim == "CPU EBI" and conf.chemFile != "monarch_cb05":
            print("Error: Set conf.chemFile = monarch_cb05 to run CPU EBI")
            raise Exception

    for i, mpiProcesses in enumerate(conf.mpiProcessesCaseOptimList):
        for j, cellsProcesses in enumerate(conf.cells):
            nCells = int(cellsProcesses / mpiProcesses)
            if nCells == 0:
                print("WARNING: Configured less cells than MPI processes, setting 1 cell per process")
                conf.mpiProcessesCaseOptimList[i] = cellsProcesses

    print("run_diffCells start")
    run_diffCells(conf)

    if get_is_sbatch() is False:
        plot_cases(conf)


if __name__ == "__main__":
    all_timesteps()
    # sns.set_theme(style="darkgrid")
    # tips = sns.load_dataset("tips")
    # ax = sns.pointplot(x="time", y="total_bill", hue="smoker", data=tips)