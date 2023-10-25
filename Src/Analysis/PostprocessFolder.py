
import numpy as np
import os
dirToAnalyse = 'Result'
dirFiles = dir(dirToAnalyse)
for numDir in np.arange(3,len(dirFiles)+1).reshape(-1):
    if not os.path.exist(str(fullfile(dirFiles(numDir).folder,dirFiles(numDir).name,'Cells'))) :
        print(fullfile(dirFiles(numDir).folder,dirFiles(numDir).name))
        infoFiles = dir(fullfile(dirFiles(numDir).folder,dirFiles(numDir).name,'/status*'))
        if len(infoFiles)==0:
            continue
        __,indices = sortrows(vertcat(infoFiles.date))
        for numT in np.transpose(indices).reshape(-1):
            scipy.io.loadmat(fullfile(dirFiles(numDir).folder,dirFiles(numDir).name,infoFiles(numT).name))
            Set.VTK = True
            PostProcessingVTK(Geo,Geo_0,Set,numStep)
