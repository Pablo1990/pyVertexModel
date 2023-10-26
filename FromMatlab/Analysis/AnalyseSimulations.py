import numpy as np
dirToAnalyse = 'Result/Relevant/Tested'
dirFiles = dir(dirToAnalyse)
for numDir in np.arange(3,len(dirFiles)+1).reshape(-1):
    allFeatures[numDir] = AnalyseSimulation(fullfile(dirFiles(numDir).folder,dirFiles(numDir).name))
