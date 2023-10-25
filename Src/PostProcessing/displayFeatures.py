import numpy as np
    
def displayFeatures(Geo = None,features = None,features0 = None,c = None,featuresToDisplay = None): 
    #DISPLAYFEATURES Summary of this function goes here
#   Detailed explanation goes here
    measurementsToDisplay_Header = struct()
    measurementsToDisplay = struct()
    for feature in np.transpose(featuresToDisplay).reshape(-1):
        numTris = 1
        setattr(measurementsToDisplay_Header,feature[0],'SCALARS ' + feature[0] + ' double\n')
        setattr(measurementsToDisplay_Header,feature[0],getattr(measurementsToDisplay_Header,(feature[0])) + 'LOOKUP_TABLE default\n')
        for f in np.arange(1,len(Geo.Cells(c).Faces)+1).reshape(-1):
            #Divide by location when required
            if endsWith(feature[0],'ByLocation'):
                baseFeature = feature[0].replace('ByLocation','')
                currentFeature = strcat(baseFeature,'_',string(Geo.Cells(c).Faces(f).InterfaceType))
                if np.all(not contains(featuresToDisplay,currentFeature) ):
                    currentFeature = baseFeature
            else:
                currentFeature = feature[0]
            for t in np.arange(1,len(Geo.Cells(c).Faces(f).Tris)+1).reshape(-1):
                #Print the values of the feature regarding the
#triangle/edge
                if contains(feature[0],'Tilting') or contains(feature[0],'Neighbours') or contains(feature[0],'Tris') or len(features0)==0:
                    result = getattr(features(numTris),(currentFeature))
                else:
                    result = getattr(features(numTris),(currentFeature))
                if isfield(measurementsToDisplay,feature[0]):
                    setattr(measurementsToDisplay,feature[0],getattr(measurementsToDisplay,(feature[0])) + sprintf('%.20f\n',result))
                else:
                    setattr(measurementsToDisplay,feature[0],sprintf('%.20f\n',result))
                numTris = numTris + 1
    
    return measurementsToDisplay_Header,measurementsToDisplay
    
    return measurementsToDisplay_Header,measurementsToDisplay