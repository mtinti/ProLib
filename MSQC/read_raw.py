def extract_dict(indata):
    
    import traceback
    import MSFileReader
    import pickle

    def save_obj(obj, name):
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL) 


    raw_file, report_name = indata[0], indata[1]
    
    res = {}
    try:
        rawfile = MSFileReader.ThermoRawfile(raw_file)
    except:
         
        return traceback.print_exc()

    
    try:
        res['GetInstrumentDescription']=rawfile.GetInstrumentDescription()
    except:
        res['GetInstrumentDescription']='failed'
    
    try:
        res['GetInstrumentID']=rawfile.GetInstrumentID()
    except:
        res['GetInstrumentID']='failed'
        
    try:
        res['GetInstSerialNumber']=rawfile.GetInstSerialNumber()
    except:
        res['GetInstSerialNumber']='failed' 
        
    try:
        res['GetInstName']=rawfile.GetInstName()
    except:
        res['GetInstName']='failed' 

    try:
        res['GetInstModel']=rawfile.GetInstModel()
    except:
        res['GetInstModel']='failed'        


    try:
        res['GetTuneData0']=rawfile.GetTuneData(0)
    except:
        res['GetTuneData0']='failed'
    

    try:
        n_methods = rawfile.GetNumInstMethods()
    except:
        n_methods = 0
    
    for i in range(n_methods):
        try:
            res['GetNumInstMethods_'+str(i)] = rawfile.GetInstMethod(i)
        except:
            pass

    rawfile.Close() 

    save_obj(res, report_name)    
    return 1