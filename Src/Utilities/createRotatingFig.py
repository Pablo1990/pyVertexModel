import matplotlib.pyplot as plt
import numpy as np
    
def createRotatingFig(fileName = None): 
    #CREATEROTATINGFIG Summary of this function goes here
#   Detailed explanation goes here
    
    #   You can use it like:
#   files = dir('/Users/pablovm/Dropbox (UCL)/TetsToShowIntercalation/*.fig')
#   for numFile = 1:length(files)
#       uiopen(fullfile(files(numFile).folder, files(numFile).name), 1)
#       createRotatingFig(fullfile(files(numFile).folder, files(numFile).name))
#   end
    
    v = VideoWriter(fileName,'MPEG-4')
    
    v.Quality = 30
    open_(v)
    plt.axis('equal')
    #lighting flat
    material('dull')
    plt.axis('off')
    for k in np.arange(1,360+1).reshape(-1):
        camorbit(1,0,'data',np.array([0,0,1]))
        frame = getframe(gcf)
        writeVideo(v,frame)
    
    close_(v)
    return
    