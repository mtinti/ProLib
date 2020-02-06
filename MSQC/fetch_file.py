def fetch_from_url(indata):
    import requests
    import os
    import MSFileReader
    import pickle
    import traceback

    def save_obj(obj, name ):
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)    
    
    
    def save_file(base_url, raw_file, save_to):
        
        url = ' http://'+base_url+'/'+raw_file
        save_to = os.path.join(save_to, raw_file)
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(save_to, 'wb') as f:
                for index,chunk in enumerate(r):
                    f.write(chunk)
    
    base_url, raw_file, save_to = indata[0],indata[1], indata[2]
    print(base_url, raw_file, save_to)
    report = os.path.join( 'E:',  os.sep, 'txt-001PTM', 'pymsfilereader', 'metadata',  raw_file+'.pkl')
    save_to = os.path.join('D:', os.sep, save_to)
    try:
        save_file(base_url, raw_file, save_to)
        rawfile = MSFileReader.ThermoRawfile(os.path.join(save_to, raw_file))
    except Exception:
        traceback.print_exc()
        if os.path.isfile(os.path.join(save_to, raw_file)):
            os.remove(os.path.join(save_to, raw_file))
        open(os.path.join('E:',  os.sep, 'txt-001PTM', 'pymsfilereader', 'failed', raw_file), 'w')
        return 0
            
            
    
    

    #except:
        #try:
            #save_file(indata)
            #rawfile = MSFileReader.ThermoRawfile(save_to)
        #except:
            #
            
           
    res = {
    'Version': rawfile.Version(),
    'GetFileName': rawfile.GetFileName(),
    'GetCreatorID': rawfile.GetCreatorID(),
    'GetVersionNumber': rawfile.GetVersionNumber(),
    'GetCreationDate': rawfile.GetCreationDate(),
    'IsError': rawfile.IsError(),
    'IsNewFile': rawfile.IsNewFile(),
    'IsThereMSData': rawfile.IsThereMSData(),
    'HasExpMethod': rawfile.HasExpMethod(),
    'InAcquisition': rawfile.InAcquisition(),
    'GetErrorCode': rawfile.GetErrorCode(),
    'GetErrorMessage': rawfile.GetErrorMessage(),
    'GetWarningMessage': rawfile.GetWarningMessage(),
    'RefreshViewOfFile': rawfile.RefreshViewOfFile(),
    'GetNumberOfControllers': rawfile.GetNumberOfControllers(),
    'GetNumberOfControllersOfTypeNodevice': rawfile.GetNumberOfControllersOfType('No device'),
    'GetNumberOfControllersOfTypeMS' :rawfile.GetNumberOfControllersOfType('MS'),
    'GetNumberOfControllersOfTypeAnalog': rawfile.GetNumberOfControllersOfType('Analog'),
    'GetNumberOfControllersOfTypeADcard': rawfile.GetNumberOfControllersOfType('A/D card'),
    'GetNumberOfControllersOfTypePDA': rawfile.GetNumberOfControllersOfType('PDA'),
    'GetNumberOfControllersOfTypeUV': rawfile.GetNumberOfControllersOfType('UV'),
    'GetCurrentController': rawfile.GetCurrentController(),
    'GetExpectedRunTime': rawfile.GetExpectedRunTime(),
    'GetMaxIntegratedIntensity': rawfile.GetMaxIntegratedIntensity(),
    'GetMaxIntensity': rawfile.GetMaxIntensity(),
    'GetInletID': rawfile.GetInletID(),
    'GetErrorFlag': rawfile.GetErrorFlag(),
    'GetFlags': rawfile.GetFlags(),
    'GetAcquisitionFileName': rawfile.GetAcquisitionFileName(),
    'GetOperator':rawfile.GetOperator(),
    'GetComment1': rawfile.GetComment1(),   
    'GetComment2': rawfile.GetComment2(),
    'GetMassTolerance': rawfile.GetMassTolerance(),
    'GetInstrumentDescription': rawfile.GetInstrumentDescription(),
    'GetInstrumentID': rawfile.GetInstrumentID(),
    'GetInstSerialNumber': rawfile.GetInstSerialNumber(),
    'GetInstName': rawfile.GetInstName(),
    'GetInstModel': rawfile.GetInstModel(),
    'GetInstSoftwareVersion': rawfile.GetInstSoftwareVersion(),
    'GetInstHardwareVersion': rawfile.GetInstHardwareVersion(),
    'GetInstFlags': rawfile.GetInstFlags(),
    'GetInstNumChannelLabels': rawfile.GetInstNumChannelLabels(),
    'IsQExactive': rawfile.IsQExactive(),
    'GetVialNumber': rawfile.GetVialNumber(),
    'GetInjectionVolume': rawfile.GetInjectionVolume(),
    'GetNumInstMethods': rawfile.GetNumInstMethods(),
    'GetInstMethodNames': rawfile.GetInstMethodNames(),
    }
    
    try:
        res['GetTuneData0']=rawfile.GetTuneData(0)
    except:
        res['GetTuneData0']='none'
        
    for i in range(rawfile.GetNumInstMethods()):
        try:
            res['GetNumInstMethods_'+str(i)] = rawfile.GetInstMethod(i)
        except:
            res['GetNumInstMethods_'+str(i)] = ''
             
    save_obj(res, report)             
    rawfile.Close()
    if os.path.isfile(os.path.join(save_to, raw_file)):
        os.remove(os.path.join(save_to, raw_file))
    return 1
                  

if __name__ == '__main__':
    pass