def fetch_from_url(indata):
    import requests
    import os
    import tqdm
    import traceback
    def save_file(url, save_to):
        fname = url.split('/')[-1]
        save_to = os.path.join(save_to, fname)
        
        #resHead = requests.head(url)
        #print(resHead.headers)
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(save_to, 'wb') as f:
                #.iter_content(102400)
                for chunk in tqdm.tqdm_notebook(r):
                    f.write(chunk)
    
    url, save_to = indata[0], indata[1]
    try:
        save_file(url, save_to)
    except:
        return traceback.print_exc()   
    return 1

    