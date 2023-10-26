    
def visualizeTets(tets = None,X = None): 
    #VISUALIZETETS Summary of this function goes here
#   Detailed explanation goes here
    allNodes = X
    figure
    tetramesh(tets,X)
    uniqueNodes = unique(tets)
    text(allNodes(uniqueNodes,1),allNodes(uniqueNodes,2),allNodes(uniqueNodes,3),cellfun(num2str,num2cell(uniqueNodes),'UniformOutput',False),'VerticalAlignment','bottom','HorizontalAlignment','right')
    return
    