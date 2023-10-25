

import numpy as np
import matplotlib.pyplot as plt
dirFiles = dir('Result/Relevant/')
numDir = 13
woundedFeaturesOnly = np.array([])
edgeLength_evo = []
timePoints = []
infoFiles = dir(fullfile(dirFiles(numDir).folder,dirFiles(numDir).name,'/status*'))
scipy.io.loadmat(fullfile(dirFiles(numDir).folder,dirFiles(numDir).name,'status1.mat'),'Set')
for numT in np.arange(3,len(infoFiles)+1).reshape(-1):
    scipy.io.loadmat(fullfile(dirFiles(numDir).folder,dirFiles(numDir).name,strcat('status',num2str(numT),'.mat')),'Geo','debris_Features','t')
    if len(debris_Features) > 0:
        currentFeatures = debris_Features[0]
        woundedFeaturesOnly[end() + 1] = currentFeatures.Tilting
        timePoints[end() + 1] = t
        edgeLength_evo[end() + 1,np.arange[1,2+1]] = np.array([Geo.Cells(4).Faces(end() - 4).Tris(4).EdgeLength,t])
    debris_Features = []

## Edge length evolution

# PEAK OF RECOLING IS AT 6 SECONDS. THUS, THAT SHOULD BE THE TIME THAT
# IT TAKES TO REACT (DELAY?).
weights = np.ones((1,1))
weights[np.arange[end() + 1,end() + 12+1]] = 0
purseString_theory = weighted_moving_average(edgeLength_evo(:,1) / edgeLength_evo(1,1),weights,10)
purseString_theory = purseString_theory ** 4.5

timePoints_norm = timePoints - timePoints(1)
#plot(0:3:60, [0.45 0.53 0.76 1.15 1.28 1.22 1.38 1.33 1.28 1.4 1.25 1.298 1.45 1.31 1.29 1.42 1.31 1.41 1.42 1.37 1.28]); hold on,
figure
plt.plot(np.arange(0,60+3,3),np.array([1,0.96,1.087,1.74,2.37,2.61,2.487,2.536,2.46,2.52,2.606,2.456,2.387,2.52,2.31,2.328,2.134,2.07,2.055,1.9,1.9]))
hold('on')
plt.plot(timePoints_norm,purseString_theory)
## Tilting
tilting = np.array([woundedFeaturesOnly[:]])
tiltingNormalised = tilting - tilting(1)
figure
plt.plot(timePoints - timePoints(1),tiltingNormalised)
hold('on')
changeInTilting = np.abs(tilting(np.arange(2,end()+1)) - tilting(np.arange(1,end() - 1+1)))
changeIntiltingNormalised = changeInTilting + 0.45 - changeInTilting(1)
changeIntiltingAdded = []
for i in np.arange(1,len(changeInTilting)+1).reshape(-1):
    changeIntiltingAdded[i] = sum(changeInTilting(np.arange(1,i+1)))

# Get values for a gaussian and then input that to conv
figure
plt.plot(timePoints(np.arange(1,end()+1)) - timePoints(1),tiltingNormalised + 0.45)
hold('on')
plt.plot(np.arange(0,60+3,3),np.array([0.45,0.53,0.76,1.15,1.28,1.22,1.38,1.33,1.28,1.4,1.25,1.298,1.45,1.31,1.29,1.42,1.31,1.41,1.42,1.37,1.28]))
plt.plot(np.arange(0,60+3,3),np.array([1,0.96,1.087,1.74,2.37,2.61,2.487,2.536,2.46,2.52,2.606,2.456,2.387,2.52,2.31,2.328,2.134,2.07,2.055,1.9,1.9]))
weights = np.array([16384,8192,4096,2048,1024,512,256,128,64,32,16,8,4,2,1])
plt.plot(timePoints(np.arange(1,end()+1)) - timePoints(1),weighted_moving_average(tiltingNormalised,weights,100000) + 0.45)
plt.legend(np.array(['tiltingNormalised','lateralCablesMaxIntensity','PurseStringIntensity','TiltingResponseDelayed_7.5secs']))
tiltingSmooth = smoothdata(tilting,'movmedian',5)
plt.plot(timePoints(np.arange(1,end()+1)) - timePoints(1),tiltingSmooth)
plt.plot(timePoints(np.arange(1,end()+1)) - timePoints(1),tiltingSmooth)
plt.plot(timePoints(np.arange(2,end()+1)) - timePoints(1),changeIntiltingAdded)
plt.plot(np.arange(0,60+3,3),np.array([0.45,0.53,0.76,1.15,1.28,1.22,1.38,1.33,1.28,1.4,1.25,1.298,1.45,1.31,1.29,1.42,1.31,1.41,1.42,1.37,1.28]))
plt.plot(np.arange(0,60+3,3),np.array([1,0.96,1.087,1.74,2.37,2.61,2.487,2.536,2.46,2.52,2.606,2.456,2.387,2.52,2.31,2.328,2.134,2.07,2.055,1.9,1.9]))
plt.legend(np.array(['tiltingNormalised','tiltingSmooth','changeInTiltingAdded','lateralCablesMaxIntensity','PurseStringIntensity']))
plt.ylim(gca,np.array([0,4]))