import os
    
def InitiateOutputFolder(Set = None): 
    # Creates Output Folder and deletes old files if exsit
    DirOutput = fullfile(pwd,Set.OutputFolder)
    if os.path.exist(str(DirOutput)):
        # 		dlt=input('Remove everything from output directory?[y]');
        dlt = 'y'
        if len(dlt)==0 or dlt == 'y':
            try:
                rmdir(DirOutput,'s')
                mkdir(DirOutput)
            finally:
                pass
    else:
        mkdir(DirOutput)
    
    Set.log = fullfile(DirOutput,Set.log)
    return Set
    
    return Set