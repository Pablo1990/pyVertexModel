import numpy as np
import os
import matplotlib.pyplot as plt
import numpy.matlib
    
def AnalyseSimulation(inputDir = None): 
    #ANALYSESIMULATION Summary of this function goes here
#   Detailed explanation goes here
    
    infoFiles = dir(fullfile(inputDir,'/status*'))
    ## Write individual results
    nonDebris_Features_time = np.array([])
    debris_Features_time = np.array([])
    wound_Features_time = np.array([])
    timePoints_nonDebris = []
    timePoints_debris = []
    beforeWounding_wound = []
    allFeatures = []
    if len(infoFiles)==0:
        print('No files!')
        return allFeatures
    
    outputDir = 'Analysis'
    mkdir(fullfile(inputDir,outputDir))
    __,indices = sortrows(vertcat(infoFiles.date))
    scipy.io.loadmat(fullfile(inputDir,infoFiles(indices(1)).name),'Set','Geo')
    cellsToAblate = Geo.cellsToAblate
    if not os.path.exist(str(fullfile(inputDir,outputDir,'info.mat'))) :
        for numT in np.transpose(indices).reshape(-1):
            scipy.io.loadmat(fullfile(inputDir,infoFiles(numT).name),'Geo','t')
            nonDeadCells = np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID])
            debrisCells = nonDeadCells(np.array([Geo.Cells(nonDeadCells).AliveStatus]) == 0)
            nonDebrisCells = nonDeadCells(np.array([Geo.Cells(nonDeadCells).AliveStatus]) == 1)
            nonDebris_Features = np.array([])
            for c in nonDebrisCells.reshape(-1):
                nonDebris_Features[end() + 1] = AnalyseCell(Geo,c)
            nonDebris_Features_table = struct2table(vertcat(nonDebris_Features[:]))
            debris_Features = np.array([])
            for c in debrisCells.reshape(-1):
                debris_Features[end() + 1] = AnalyseCell(Geo,c)
            if not len(debris_Features)==0 :
                debris_Features_time[end() + 1] = struct2table(np.array([debris_Features[:]]))
                wound_features = ComputeWoundFeatures(Geo)
                wound_Features_time[end() + 1] = struct2table(ComputeFeaturesPerRow(Geo,cellsToAblate,wound_features))
                #writetable(debris_Features_time{end}, fullfile(inputDir, outputDir, strcat('debris_features_', num2str(t),'.csv')))
                timePoints_debris[end() + 1] = t
            else:
                beforeWounding_debris = nonDebris_Features_table(ismember(nonDebris_Features_table.ID,cellsToAblate),:)
                beforeWounding_nonDebris = nonDebris_Features_table(not ismember(nonDebris_Features_table.ID,cellsToAblate) ,:)
                beforeWounding_wound = ComputeWoundFeatures(Geo,cellsToAblate)
                beforeWounding_wound = struct2table(ComputeFeaturesPerRow(Geo,cellsToAblate,beforeWounding_wound))
            nonDebris_Features_time[end() + 1] = nonDebris_Features_table
            timePoints_nonDebris[end() + 1] = t
            #writetable(nonDebris_Features_table, fullfile(inputDir, outputDir, strcat('cell_features_', num2str(t),'.csv')))
            clearvars('wound_features')
        writetable(vertcat(wound_Features_time[:]),fullfile(inputDir,outputDir,strcat('wound_features.csv')))
        save(fullfile(inputDir,outputDir,'info.mat'),'beforeWounding_debris','timePoints_nonDebris','nonDebris_Features_time','beforeWounding_wound','beforeWounding_nonDebris','timePoints_debris','wound_Features_time','debris_Features_time')
    else:
        scipy.io.loadmat(fullfile(inputDir,outputDir,'info.mat'))
    
    if len(wound_Features_time) > 1:
        ## Write summary results with the following features:
# Wound: area (apical, basal), volume.
# Cells at the wound edge: cell height, n number, tilting, volume,
# intercalations, number of neighbours (3D, apical, basal), area (apical
# and basal).
# Cells not at the wound edge: same features as before.
        initialWound_features_sum = sum(table2array(beforeWounding_debris))
        initialWound_features_avg = mean(table2array(beforeWounding_debris))
        initialWound_features_std = std(table2array(beforeWounding_debris))
        initialCells_features_sum = sum(table2array(beforeWounding_nonDebris))
        initialCells_features_avg = mean(table2array(beforeWounding_nonDebris))
        initialCells_features_std = std(table2array(beforeWounding_nonDebris))
        ## Features at timepoint N after wounding.
        nonDebris_Features = np.array([])
        cells_features_sum = array2table(initialCells_features_sum,'VariableNames',cellfun(lambda x = None: strcat('sum_',x),beforeWounding_nonDebris.Properties.VariableNames,'UniformOutput',False))
        cells_features_avg = array2table(initialCells_features_avg,'VariableNames',cellfun(lambda x = None: strcat('avg_',x),beforeWounding_nonDebris.Properties.VariableNames,'UniformOutput',False))
        cells_features_std = array2table(initialCells_features_std,'VariableNames',cellfun(lambda x = None: strcat('std_',x),beforeWounding_nonDebris.Properties.VariableNames,'UniformOutput',False))
        wound_Features = beforeWounding_wound
        for numTime in np.arange(1,60+1).reshape(-1):
            if numTime > timePoints_nonDebris(end()):
                numTime = numTime - 1
                break
            nonDebris_Features[numTime] = distanceTime_Features(Set,timePoints_nonDebris,nonDebris_Features_time,numTime)
            #debris_Features{numTime} = distanceTime_Features(Set, timePoints_debris, debris_Features_time, numTime);
            cells_features_sum[end() + 1,:] = array2table(sum(table2array(nonDebris_Features[numTime])))
            cells_features_avg[end() + 1,:] = array2table(mean(table2array(nonDebris_Features[numTime])))
            cells_features_std[end() + 1,:] = array2table(std(table2array(nonDebris_Features[numTime])))
            wound_Features[end() + 1,:] = array2table(table2array(distanceTime_Features(Set,timePoints_debris,wound_Features_time,numTime)))
        allFeatures = np.array([cells_features_sum,cells_features_avg,cells_features_std,wound_Features])
        allFeatures.time = np.transpose(np.array([np.arange(0,numTime+1)]))
        writetable(allFeatures,fullfile(inputDir,outputDir,strcat('cell_features.csv')))
        ## Figure of features evolution.
        woundedFeaturesOnly = table2array(allFeatures)
        rowVariablesIds = allFeatures.Properties.VariableNames(cellfun(lambda x = None: contains(x,'Row1'),allFeatures.Properties.VariableNames))
        featuresWithRowsIds = cellfun(lambda x = None: x.replace('Row1_',''),rowVariablesIds,'UniformOutput',False)
        nonRowVariablesIds = find(cellfun(lambda x = None: np.logical_or(np.logical_or(contains(x,'sum'),contains(x,'avg')),contains(x,'std')),allFeatures.Properties.VariableNames))
        for numColumn in nonRowVariablesIds.reshape(-1):
            plt.figure('WindowState','maximized','Visible','off')
            ax_all = axes
            x = woundedFeaturesOnly(:,end())
            y = woundedFeaturesOnly(:,numColumn)
            plt.plot(x,y)
            lgd = plt.legend(allFeatures.Properties.VariableNames(numColumn),'FontSize',6)
            plt.xlim(ax_all,np.array([0,60]))
            plt.xlabel(ax_all,'time')
            saveas(ax_all,fullfile(inputDir,outputDir,strcat(allFeatures.Properties.VariableNames[numColumn],'.png')))
            plt.legend(ax_all,'hide')
            saveas(ax_all,fullfile(inputDir,outputDir,strcat(allFeatures.Properties.VariableNames[numColumn],'_noLegend.png')))
            close_('all')
        ## Figure of features evolution to overlap with others
        for nameFeature in featuresWithRowsIds.reshape(-1):
            plt.figure('WindowState','maximized','Visible','off')
            ax_all = axes
            hold('on')
            rowVariablesIds = find(cellfun(lambda x = None: np.logical_and(np.logical_and(np.logical_and(endsWith(x,nameFeature[0]),not contains(x,'sum') ),not contains(x,'avg') ),not contains(x,'std') ),allFeatures.Properties.VariableNames))
            for numColumn in rowVariablesIds.reshape(-1):
                x = np.transpose(woundedFeaturesOnly(:,end()))
                y = np.array([woundedFeaturesOnly(:,numColumn)]) / woundedFeaturesOnly(1,numColumn)
                #y(2) to analyse steep correlation to Set variables
                xx = np.array([[x],[x]])
                y = np.transpose(y)
                yy = np.array([[y],[y]])
                zz = np.zeros((xx.shape,xx.shape))
                cc = np.matlib.repmat(numColumn,yy.shape)
                surf(ax_all,xx,yy,zz,cc,'EdgeColor','interp','LineWidth',4)
            lgd = plt.legend(allFeatures.Properties.VariableNames(rowVariablesIds),'FontSize',6)
            lgd.NumColumns = 2
            ylimAxis = plt.ylim(ax_all)
            plt.ylim(ax_all,np.array([0,ylimAxis(2)]))
            plt.xlim(ax_all,np.array([0,60]))
            plt.xlabel(ax_all,'time')
            plt.ylabel(ax_all,'Change')
            saveas(ax_all,fullfile(inputDir,outputDir,strcat(nameFeature[0],'Evolution.png')))
            plt.legend(ax_all,'hide')
            saveas(ax_all,fullfile(inputDir,outputDir,strcat(nameFeature[0],'Evolution_noLegend.png')))
            close_('all')
    
    return allFeatures
    
    return allFeatures