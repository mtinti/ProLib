#load library

import sys
import os 



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns
from tqdm import tqdm_notebook
import gc
from adjustText import adjust_text
from matplotlib.lines import Line2D
import os
from multiprocessing import Pool
import urllib.request
from bs4 import BeautifulSoup
import re
import time
import math
from functools import wraps   
import itertools
from itertools import (takewhile,repeat)
from io import StringIO
import MSFileReader
import deepdiff
import json
import pprint
import pickle


#functions to compare two raw files methods
def make_method_dictionary(indata):
    res = {}
    tag = 'OVERALL'
    counts = []
    for n in indata.split('\n'):
        key = n.split('  ')[0].strip()
        if len(n.split('  '))>1 and len(key)>0:
            counts.append(key)
    #print(counts)
    for n in indata.split('\n'):
        key = n.split('  ')[0].strip()
        value = n.split('  ')[-1].strip()
        if len(n.split('  '))>1 and len(key)>0:        
            if key in res:
                res[key+' '+str(counts.count(key))]=value
                counts.pop(counts.index(key))
            else:
                res[key]=value
    return res




def get_method(raw_file_path, verbose = 0):
    if raw_file_path.lower().endswith('.raw'):
        rawfile = MSFileReader.ThermoRawfile(raw_file_path)
        if verbose > 0:
            print('finding method from:',rawfile.GetInstModel())
        n_methods = rawfile.GetNumInstMethods()
        texts = [rawfile.GetInstMethod(i) for i in range(n_methods)]
    
    elif raw_file_path.lower().endswith('.pkl'):
        temp_dict = pickle.load(open( raw_file_path, "rb" ) )
        if verbose > 0:
            print('finding method from:',temp_dict['GetInstModel'])
        keys = [n for n in temp_dict.keys() if 'GetNumInstMethods_' in n]
        texts = [temp_dict[k] for k in keys]
    else:
        Exception('no method found 1')
    for text in texts:
        #text = rawfile.GetInstMethod(i)
        if 'Method of' in text:
            return text
    raise Exception('no method found 2')


def find_method_diff(raw_file_path_1, raw_file_path_2):
    dict_1 = make_method_dictionary(get_method(raw_file_path_1)) 
    dict_2 = make_method_dictionary(get_method(raw_file_path_2))
    diff = deepdiff.DeepDiff(dict_1, dict_2)
    pp = pprint.PrettyPrinter(indent=10)
    for key in diff.keys():
        print('_______',key,'_______')
        pp.pprint(diff[key])
        print('_______')

#functions to extract gradient from raw file Methods
#our in house Methods
def process_1(text):
    start = 0
    for index, line in enumerate(text.split('\n')):
        if '0.000 [min] Start Run' in line:
            start = index
            break
    minute = 0
    b_value = 0
    minutes = []
    b_values = []
    for line in text.split('\n')[start:]:
        #print (line)
        if '[min]' in line:
            #print (line.split())
            minute = line.split(' ')[0]
            
        if 'PumpModule.NC_Pump.%B.Value' in line:
            b_value = line.split(' ')[-2]
            b_values.append(b_value)
            minutes.append(minute)
            #print(minute, b_value)
            #print('ok')
        
    #print('minutes',minutes)
    #print('b_values',b_values)       
    #print(len(minutes),len(b_values))
    return (minutes,b_values)

#most of the cases            
def process_2(text):
    limits=[]
    for index, line in enumerate(text.split('\n')):
        if 'Gradient:' in line:
            start=index
        if 'Pre-column' in line:
            end=index   
    #print(start,end)
    minutes = []
    b_values = []
    for index, line in  enumerate(text.split('\n')[start+2:end-1]):
            #text+=line
            item_list = line.split()
            minutes.append(item_list[0].split(':')[0])
            b_values.append(item_list[-1])
    #print(minutes,b_values)        
    #print(len(minutes),len(b_values))       
    return (minutes,b_values)       

def process_3(text):
    minutes = []
    b_values = []
    list_text = text.split('\n')
    for idex,n in enumerate(list_text):
        if ('Pump' in n) and ('Flow' in n):
            if len(n.split())==5:
                minutes.append(n.split()[0])
                b_values.append(list_text[idex+1].split()[2])
    return (minutes,b_values) 


#make datapoints by minutes
def expand_gradient(minutes,b_values):
    #e_minutes, e_bvalues=[],[]
    #c_bval = b_values[0]
    #print(minutes)
    x= np.arange(minutes[0], minutes[-1], 1)
    
    y = np.interp(x, minutes, b_values)
    #for n in np.arange(minutes[0], minutes[-1], 1):
    #    if n in minutes:
    #        c_bval =  b_values[minutes.index(n)]
    #    e_minutes.append(n)
    #    e_bvalues.append(c_bval)
    return (x,y)



#controller
def extract_gradient(in_path):
    if in_path.lower().endswith('.raw'):
        rawfile = MSFileReader.ThermoRawfile(in_path)  
        n_methods = rawfile.GetNumInstMethods()
        n_methods = [rawfile.GetInstMethod(i) for i in range(n_methods)]
    elif in_path.lower().endswith('.pkl'):
        temp_dict = pickle.load(open( in_path, "rb" ) )
        keys = [n for n in temp_dict.keys() if 'GetNumInstMethods_' in n]
        n_methods = [temp_dict[k] for k in keys]
    else:
        Exception('no method found 1')    
    
    found = 0
    for text in n_methods:
        #print(text)
        
        #print(i)
        if 'Gradient:' in text:
            #print('run 1')
            minutes, b_values=process_2(text)
            found+=1
        elif '---- Script ----' in text:
            #print('run 2')
            #print(text)
            minutes ,b_values=process_1(text)
            found+=1
            #print(text)
        elif 'Dionex Chromatography' in  text:
            #print('run 3')
            minutes ,b_values=process_3(text)
            found+=1
            
    if found ==0:
        print('Gradient processing not implemented')
        return(0,0)
    if len(minutes) != len(b_values):
        raise Exception('len is different')
    minutes = [int(float(n)) for n in minutes]
    b_values = [int(float(n)) for n in b_values]
    minutes, b_values = expand_gradient(minutes,b_values)
    return (minutes,b_values)



#function to visualize QC metric for a raw file
#Functions for QC plots
def __plot_injection_time(ax, tempMSdf, tempMSMSdf, tempMSMSMSdf):
    tempMSMSdf.groupby('RT_round').mean().plot(
        kind='line',
        x='Retention time',
        y='Ion injection time',
        ax=ax,
        label='mean MSMS')
    if len(tempMSdf) > 0:
        tempMSdf.groupby('RT_round').median().plot(
        kind='line',
        x='Retention time',
        y='Ion injection time',
        ax=ax,
        label='median MS')

    if len(tempMSMSMSdf) > 0:
        tempMSMSMSdf.groupby('RT_round').median().plot(
        kind='line',
        x='Retention time',
        y='Ion injection time',
        ax=ax,
        label='median MS3')

    
    ax.set_ylabel('Millisec')
    ax.set_title('QC: Ion Injection Time')
    ax.legend()
        
def __plot_gradient(ax, tempMSMSdf, gradient):
    
    #print(tempMSMSdf.head())
    #print(tempMSMSdf['Retention time'].head())
    bins = int(tempMSMSdf['Retention time'].max()+1)
    #print(tempMSMSdf['Retention time'][tempMSMSdf['Identified']=='-'].head())
    temp = tempMSMSdf[ tempMSMSdf['Identified']=='+' ].groupby('RT_round')['Retention time'].size()
    ax.plot(temp.index.values,temp.values,'g-',label='Identified')
    
    temp = tempMSMSdf[ tempMSMSdf['Identified']=='-' ].groupby('RT_round')['Retention time'].size()
    ax.plot(temp.index.values,temp.values,'r-',label='not identified')
    
    #print(tempMSMSdf.groupby('RT_round')['Retention time'].size().head())
    
    '''
    tempMSMSdf['Retention time'][tempMSMSdf['Identified']=='-'].plot(
        kind='hist',
        histtype='step', 
        bins=bins, 
        color='b',
        label='not identified',
        ax=ax,
     linewidth =1.2)
    
    tempMSMSdf['Retention time'][tempMSMSdf['Identified']=='+'].plot(
        kind='hist',
        histtype='step', 
        bins=bins, 
        color='r',
        label='identified',
        ax=ax,
        linewidth =1.2)
    '''
    ax.set_ylabel('Count')
    
    if len(gradient) >0:
        ax2 = ax.twinx()
        ax2.plot(gradient[0],gradient[1],'g--',alpha=0.5, label='Acetonitrile')
        ax2.set_ylabel('Acetonitrile %', fontsize=16)
        ax2.grid(False)
        ax2.set_yticks(np.arange(5, 100, 10))
        ax2.legend(loc='upper center', bbox_to_anchor=(0.95, -0.02))
    
    #handles, labels = ax.get_legend_handles_labels()

        
    #new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    ax.legend(#handles=new_handles, labels=labels,
              loc='upper center', bbox_to_anchor=(0.1, -0.02))
    ax.set_title('QC: Chromatography Efficency')
  
def __plot_ion_current(ax, tempMSdf, tempMSMSdf, tempMSMSMSdf):
    
    tempMSMSdf = tempMSMSdf.groupby('RT_round').median()
    tempMSMSdf['Total ion current']=np.log10(tempMSMSdf['Total ion current'])
    tempMSMSdf.plot(kind='line',
                    x='Retention time',
                    y='Total ion current',
                    ax=ax,
                    label='MSMS')
    if len(tempMSdf) > 0:
        tempMSdf =tempMSdf.groupby('RT_round').median()
        tempMSdf['Total ion current']=np.log10(tempMSdf['Total ion current'])
        tempMSdf.groupby('RT_round').median().plot(
        kind='line',
        x='Retention time',
        y='Total ion current',
        ax=ax,
        label='MS')
    
    if len(tempMSMSMSdf) > 0:
        tempMSMSMSdf =tempMSMSMSdf.groupby('RT_round').median()
        tempMSMSMSdf['Total ion current']=np.log10(tempMSMSMSdf['Total ion current'])
        tempMSMSMSdf.groupby('RT_round').median().plot(
        kind='line',
        x='Retention time',
        y='Total ion current',
        ax=ax,
        label='MS3')
     
    ax.set_ylabel('LOG10 Ion Current')
    ax.set_title('QC: Instrument Operativity')

def __plot_cycle_time(ax, tempMSdf):
    if len(tempMSdf) > 0:
        tempMSdf.groupby('RT_round').median().plot(
        kind='line',
        x='Retention time',
        y='Cycle time',
        label='MS',
        ax = ax)
    ax.set_ylabel('Cycle time')
    ax.set_title('QC: Instrument Operativity')
    
#function to connect all the subplots
def plot_raw_file(tempMSdf, tempMSMSdf,  tempMSMSMSdf, title, gradient=()):
    fig,axes=plt.subplots(figsize=(16,10), ncols=2, 
    nrows=2,  sharex=True)
    #just to be sure
    if len(tempMSdf) > 0:
        #print('ok')
        tempMSdf = tempMSdf.sort_values('Retention time')
    #print(tempMSMSdf.head())
    
    
    tempMSMSdf = tempMSMSdf.sort_values('Retention time')
    
    __plot_gradient(axes[0,0],  tempMSMSdf, gradient=gradient)
    __plot_injection_time(axes[0,1], tempMSdf, tempMSMSdf, tempMSMSMSdf)
    __plot_ion_current(axes[1,0], tempMSdf, tempMSMSdf, tempMSMSMSdf)
    __plot_cycle_time(axes[1,1], tempMSdf)
    for a,b in [(0,0),(0,1),(1,0),(1,1)]:
        ax = axes[a, b]
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(16)
    return fig,axes 
    
#make a safe file name
def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = re.sub('[^\\w\\s-]', '', value).strip().lower()
    value = re.sub('[-\\s]+', '-', value)
    return value

def compare_metric(df_summary, all_the_others, col, TXT_PATH):
    fig,ax = plt.subplots(figsize=(6,6))
    temp = pd.concat([all_the_others[col],df_summary[col]]).to_frame()
    temp['facility']=['others' for n in all_the_others.index.values]+['this' for n in df_summary.index.values ]
    #print (temp.tail())
    sns.boxplot(data=temp, y=col, x='facility',showfliers=False)
    ax.set_title('QC: '+col,fontsize=20)
    ax.set_xlabel('Experiment', fontsize=20)
    ax.set_ylabel(col,fontsize=20)
    plt.legend([])
    plt.savefig(os.path.join(TXT_PATH, slugify(col)+'.png'))
    plt.show()    


def fractionation_report(msmsIdentified, axes, TXT_PATH):
    #identify unique sequences in each raw file
    unique_pep = msmsIdentified.groupby('Raw file')['Sequence'].unique()
    #print(unique_pep)
    res_perc = []
    res_unique = []
    tot = []
    #for each set of unique sequences in a raw file
    for index, n in enumerate(unique_pep):
        #print(index)
        #grab all the other peptides
        other = [n for n in range(len(unique_pep)) if n != index]
        
        #identify data for unique peptide
        raw_file = set(unique_pep.iloc[index])
        #print(raw_file)
        
        #create a set with all the peptides identified in the othe raw files
        rest = set(list(itertools.chain.from_iterable(unique_pep.iloc[other])))
        
        #number of unique peptides in a fraction
        tot_fraction = len(unique_pep.iloc[index])
        
        unique_fraction = len(raw_file)
        perc =  len(raw_file- rest)/ len(raw_file)
        res_perc.append(len(raw_file- rest)/len(raw_file))
        res_unique.append(unique_fraction)
        tot.append(tot_fraction)

    tmpdf = pd.DataFrame(index=[unique_pep.index.values])
    tmpdf['res_perc']=res_perc
    tmpdf['unique']=res_unique
    tmpdf['tot']=tot
    #tmpdf['tot'] = tmpdf['tot']-tmpdf['unique']
    tmpdf['res_perc'].plot(kind='barh',label='perc',ax=axes[0])
    axes[0].set_title('unique in fraction')
    
    tmpdf[['unique','tot']].plot(kind='barh',label='n', stacked=True, ax=axes[1])
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(TXT_PATH,'fractionation_report.png'))
    plt.show() 
  
  

#functions to download from the PRIDE web site
#the list of raw file and instument used
#extract any file deposited in pride
def _extract_files(table):
    file_names,file_links = [],[]
    for raw_index, row in enumerate(table.find_all('tr')):
        columns = row.find_all('td')
        for column_index, content in enumerate(columns):
            if column_index == 0:
                temp_file = content.get_text().strip()
                #print(temp_file)
                #if '.raw' in temp_file.lower():
                file_names.append(temp_file)
            if column_index == 2:
                #print()
                #temp_link = content.get('href')
                #print(temp_file)
                #if '.raw' in temp_file.lower():
                file_links.append(content.find_all('a')[0].get('href'))              
    return file_names,file_links

#create a table file pride ids and deposited files
def get_table(soup, pride_id):
    raw_files = []
    pride_links = []
    tables = soup.findAll('table', {'class': 'summary-table'})
    for table in tables:
        out = _extract_files(table)
        raw_files += out[0]
        pride_links += out[1]
    temp_df = pd.DataFrame()
    temp_df['raw_files'] = raw_files
    temp_df['raw_files_links'] = pride_links
    temp_df['pride_id'] = pride_id
    return temp_df

#function to extract the instrument type-model
#can be multiple type
def get_instrument(soup):
    class_='grid_12'
    tag='h5'
    text='Instrument'
    res = []
    for data in soup.find_all('div', class_=class_):
        for a in data.find_all('a'):    
            if data.find(tag) and data.find(tag).text == text:
                res.append(a.text)
    return ';'.join(res)


#function to get the lab head from
#the pride contacts 
def get_contact(soup):
    class_='grid_16 left-column'
    tag='h5'
    for data in soup.find_all('div', class_=class_):
        for p in data.find_all('p'):
            for a in p.find_all('a'): 
                if 'mailto' in str(a):
                    author = a.text
                    for n in p.text.split('\n'):
                        if n.endswith('('):
                            return(n[0:-1].strip())
    return ''


#function to extract the submission and publication dates
def get_date(soup):
    submission = ''
    publication = ''
    class_='grid_16 left-column'
    tag='h5'
    for data in soup.find_all('h5'):
        #print(data)
        if data.text == 'Submission Date':
            submission = data.next_sibling.next_sibling.text.strip()
        if data.text == 'Publication Date':
            publication = data.next_sibling.next_sibling.text.strip()
            
        #for p in data.find_all('p'):
            #for a in p.find_all('a'): 
                #if 'mailto' in str(a):
                    #author = a.text
                    #for n in p.text.split('\n'):
                        #if n.endswith('('):
                            #return(n[0:-1].strip())
    return submission,publication


#function to request a pride web page from id
def get_page(pride_id, extension=''):
    pride_page = 'https://www.ebi.ac.uk/pride/archive/projects/{pride_id}/'+extension
    response =urllib.request.urlopen(pride_page.format(pride_id=pride_id))
    html = response.read()
    soup=BeautifulSoup(html,features="lxml")
    return soup
    
#the list of PRIDE ids used in the analysis
pride_ids = [
    'PXD004373','PXD002425','PXD001060','PXD000681','PXD004817',
    'PXD003627','PXD002765','PXD001962','PXD003198','PXD006475',
    'PXD001180','PXD004357','PXD004981','PXD001550','PXD000696',
    'PXD000089','PXD000217','PXD003492','PXD004181','PXD004340',
    'PXD000901','PXD002850','PXD000523','PXD002614','PXD006055',
    'PXD001101','PXD003108','PXD000496','PXD002255','PXD003908',
    'PXD002172','PXD001546','PXD001565','PXD000462','PXD000680',
    'PXD002646','PXD002127','PXD000474','PXD001374','PXD004940',
    'PXD000612','PXD003523','PXD001333','PXD001559','PXD001305',
    'PXD004452','PXD005181','PXD001196','PXD005366','PXD000293',
    'PXD002704','PXD001603','PXD001381','PXD004252','PXD001186',
    'PXD001094','PXD002871','PXD000341','PXD001428','PXD004447',
    'PXD002055','PXD004415','PXD002023','PXD000185','PXD001129',
    'PXD002839','PXD002990','PXD000985','PXD000836','PXD002394',
    'PXD000599','PXD003660','PXD001812','PXD000275','PXD002072',
    'PXD000472','PXD001114','PXD002735','PXD001121','PXD001222',
    'PXD001739','PXD000750','PXD004442','PXD001543','PXD000451',
    'PXD000242','PXD001189','PXD001170','PXD003822','PXD000597',
    'PXD005214','PXD003115','PXD004945','PXD000970','PXD001253',
    'PXD000497','PXD003531','PXD002286','PXD000238','PXD001115',
    'PXD002635','PXD003708','PXD002496','PXD000222','PXD003712',
    'PXD002135','PXD003529','PXD000225','PXD001563','PXD001560']    

def make_pride_info():
    #These PRIDE IDs contains raw files that are deposited by company,
    #they d onot have a Laboratory Head tag
    to_skip = {
    'PXD000089':"Takashi Shiromizu, Proteome Research Center",
    'PXD000185':"Pedro Casado-Izquierdo, Cell Signalling",
    'PXD000217':"Pedro Casado-Izquierdo, Cell Signalling",
    'PXD000222':"Teck Yew Low, Utrecht Univeristy",
    'PXD000225':"Li Xia, Pathophysiology",
    'PXD000238':"Nicholas Graham, Molecular and Medical Pharmacology / Crump Institute for Molecular Imaging",
    'PXD000242':"Florian Beck, Boehringer Ingelheim",
    'PXD000341':"Mogjib SALEK, Pathology",
    'PXD000451':"Bjoern Titz, Mol & Med. Pharmacology",
    'PXD000462':"Nathan Tedford, Biological Engineering",
    'PXD000472':"Pedro Casado-Izquierdo, Cell Signalling",
    'PXD000496':"Mario Oroshi, Proteomics",
    'PXD000497':"Teck Yew Low, Utrecht Univeristy",
    'PXD001603':"Megan Chircop, Cell Cycle Unit Childrens Medical Research Institute"
    }
    #scrape pride main experiment page for:
    instrument_list = []
    lab_head_list = []
    sub_date_list = []
    pub_data_list = []
    for pride_id in tqdm_notebook(pride_ids):
        #print(pride_id)
        html = get_page(pride_id)
        instrument_list.append(get_instrument(html))
        if pride_id in to_skip:
            lab_head_list.append(to_skip[pride_id])
        else:
            lab_head_list.append(get_contact(html))
        submission,publication = get_date(html)
        sub_date_list.append(submission)
        pub_data_list.append(publication)
        #print(lab_head_list)
    #len(pride_ids),len(instrument_list)
    pride_info = pd.DataFrame()
    pride_info['pride_id']=pride_ids
    pride_info['instrument']=instrument_list
    pride_info['lab']=lab_head_list
    pride_info['sub_date_list']=sub_date_list
    pride_info['pub_data_list']=sub_date_list
    pride_info.to_csv('pride_info.csv')
    pride_info.head()


def make_pride_file_names():
    #get the file names deposited in each PRIDE id
    df_list = []
    for pride_id in tqdm_notebook(pride_ids):
        html = get_page(pride_id, extension='files')
        df_list.append(get_table(html, pride_id))
    pride_df = pd.concat(df_list)
    pride_df.to_csv('pride_files.csv')
    #check the file number by PRIDE id
    pride_df.groupby('pride_id').size().sort_values().head()
    
    
 
def timer(func):
    """This decorator prints the execution time for the decorated function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("{} ran in {}s".format(func.__name__, round(end - start, 2)))
        return result
    return wrapper

#select the smallest dtype to save space
__dtype={
    'Charge':np.uint8,
    'Ion injection time':np.float32,
    'Retention time':np.float32,
    'Raw file':str,
    'Identified':str,
    'Total ion current':np.float32,
    'Cycle time':np.float32,
    'Total ion current':np.float32,
     'Sequence':str }

__ms_columns = [
    'Raw file',
    'Retention time',
    'Cycle time',
    'Total ion current',
    'Ion injection time',    
]

__msms_columns = [
    'Charge',
    'Ion injection time',
    'Retention time',
    'Raw file',
    'Identified',
    'Total ion current',
    'Sequence'
]


__msmsms_columns = [
    'Raw file',
    'Ion injection time',
    'Retention time',
    'Total ion current',
]

__chunksize = 300000


#function to add QC metrics




def rawincount(filename):
    f = open(filename, 'rb')
    bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
    return sum( buf.count(b'\n') for buf in bufgen )

@timer
def find_chunks(infile='msScans.txt', chunksize=__chunksize):
    tot_lines = rawincount(infile)#sum(1 for i in open(infile, 'rb'))
    print(tot_lines, 'lines in', infile )
    chunksize = math.ceil(tot_lines/float(chunksize))
    print(chunksize, 'expected chunks of ', tot_lines/chunksize, 'rows')
    return chunksize

def load_chunks(infile='msScans.txt', chunksize=__chunksize,
                usecols = [], dtype = {}, tot_chunks=int):
    res = []
    for df_chunk in tqdm_notebook(pd.read_csv(
        infile,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
        sep='\t'), total=tot_chunks):
        
        res.append(df_chunk)
    
    res = pd.concat(res)
    print(res.shape)
    return res
    
#QC metric functions  
def add_max_retention_time(msScans, df_summary):
    col = 'Retention time'
    #take the max retention time oserved in the MS table
    temp = msScans.groupby('Raw file')[col].max()
    #print('stats for', col)
    #print(temp.describe())
    temp = temp.to_frame().reset_index()
    temp.columns = ['Raw file','max_retention_time']
    #atach information to the summary file
    df_summary.drop('max_retention_time', axis=1,inplace=True,errors='ignore')   
    df_summary = df_summary.merge(temp, left_on='Raw file', right_on='Raw file', how='left')
    return df_summary
    
def add_median_injection_time_ms(msScans, df_summary):
    col = 'Ion injection time'
    temp = msScans.groupby('Raw file')[col].median()
    #print('stats for', col)
    #print(temp.describe())
    temp = temp.to_frame().reset_index()
    temp.columns = ['Raw file', 'median_injection_time_ms']
    df_summary.drop('median_injection_time_ms', axis=1,inplace=True,errors='ignore')    
    df_summary = df_summary.merge(temp, left_on='Raw file', right_on='Raw file', how='left')
    return df_summary   
    
def add_median_injection_time_msms(msmsScans, df_summary):
    col = 'Ion injection time'
    temp = msmsScans.groupby('Raw file')[col].median()
    #print('stats for', col)
    #print(temp.describe())
    temp = temp.to_frame().reset_index()
    temp.columns = ['Raw file', 'median_injection_time_msms']
    df_summary.drop('median_injection_time_msms', axis=1,inplace=True,errors='ignore')  
    df_summary = df_summary.merge(temp, left_on='Raw file', right_on='Raw file', how='left')
    return df_summary
    
def add_retention_spread(msmsIdentified, df_summary):
    temp_max = msmsIdentified[msmsIdentified['RT_bin_rank_cut']==2].groupby(
        ['Raw file'])['Retention time'].max()
    temp_max = temp_max.to_frame().reset_index()
    temp_max.columns = ['Raw file', 'retention_spread']
    
    df_summary.drop('retention_spread', axis=1,inplace=True,errors='ignore')  
    df_summary = df_summary.merge(temp_max, left_on='Raw file', right_on='Raw file', how='left')
    
    df_summary.drop('norm_retention_spread', axis=1,inplace=True,errors='ignore')
    df_summary['norm_retention_spread']=(df_summary['retention_spread']*2)/df_summary['max_retention_time']
    
    df_summary.drop('abs_retention_spread', axis=1,inplace=True,errors='ignore')
    df_summary['abs_retention_spread'] = abs(1-df_summary['norm_retention_spread'])
    return df_summary

def add_retention_spread(msmsIdentified, df_summary):
    temp_max = msmsIdentified[msmsIdentified['RT_bin_rank_cut']==2].groupby(
        ['Raw file'])['Retention time'].max()
    temp_max = temp_max.to_frame().reset_index()
    temp_max.columns = ['Raw file', 'retention_spread']
    
    df_summary.drop('retention_spread', axis=1,inplace=True,errors='ignore')  
    df_summary = df_summary.merge(temp_max, left_on='Raw file', right_on='Raw file', how='left')
    
    df_summary.drop('norm_retention_spread', axis=1,inplace=True,errors='ignore')
    df_summary['norm_retention_spread']=(df_summary['retention_spread']*2)/df_summary['max_retention_time']
    
    df_summary.drop('abs_retention_spread', axis=1,inplace=True,errors='ignore')
    df_summary['abs_retention_spread'] = abs(1-df_summary['norm_retention_spread'])
    return df_summary 

def add_pep_min(msmsIdentified, df_summary):
    temp = msmsIdentified[
    #extract the middle part for each raw file
    msmsIdentified['RT_bin_rank_cut'].isin([2,3])].groupby(
    #count number of peptide in each minute
    ['Raw file','RT_round']).size().groupby(
    #take the median
    ['Raw file']).median()
    
    temp = temp.to_frame().reset_index()
    temp.columns = ['Raw file', 'peptides_per_minute']
    df_summary.drop('peptides_per_minute', axis=1,inplace=True,errors='ignore') 
    df_summary = df_summary.merge(temp, left_on='Raw file', right_on='Raw file', how='left')
    return df_summary

def add_spray_instability_msms(msmsScans, df_summary):
    temp = msmsScans[
        #extract the middle part for each raw file
        msmsScans['RT_bin_qcut'].isin([2,3])][['Raw file', 'RT_round', 'Total ion current']].groupby(
        #get the ion current column, and groupby raw file
        ['Raw file','RT_round']).median()

    temp = temp.groupby(
        ['Raw file'])['Total ion current'].apply(
        #compute percentage change
        lambda x: x.pct_change()).groupby(
        ['Raw file']).apply(
        #compute if it's greather than 10
        lambda x: x.ge(10)
        #find how many times is hover the threshold
        #apparenty, apply of apply return df
        ).groupby('Raw file').apply(lambda x: x.sum())
    
    temp = temp.to_frame().reset_index()
    temp.columns = ['Raw file', 'spray_instability_msms']
    df_summary.drop('spray_instability_msms', axis=1,inplace=True,errors='ignore') 
    df_summary = df_summary.merge(temp, left_on='Raw file', right_on='Raw file', how='left')
    return df_summary

def add_spray_instability_ms(msScans, df_summary):
    temp = msScans[
        #extract the middle part for each raw file
        msScans['RT_bin_qcut'].isin([2,3])][['Raw file','RT_round', 'Total ion current']].groupby(
        #get the ion current column, and groupby raw file an minute
        ['Raw file','RT_round']).median()
    temp = temp.groupby(
        ['Raw file'])['Total ion current'].apply(
        #compute percentage change
        lambda x: x.pct_change()).groupby(
        ['Raw file']).apply(
        #compute if it's greather than 10
        lambda x: x.ge(10)
        #find how many times is hover the threshold
        #apparenty, apply of apply return df
        ).groupby('Raw file').apply(lambda x: x.sum())

    temp = temp.to_frame().reset_index()
    temp.columns = ['Raw file', 'spray_instability_ms']
    df_summary.drop('spray_instability_ms', axis=1,inplace=True, errors='ignore') 
    df_summary = df_summary.merge(temp, left_on='Raw file', right_on='Raw file', how='left')
    return df_summary

def qc_pipline(TXT_PATH):

    df_summary= pd.read_csv(os.path.join(TXT_PATH,'summary.txt'), sep='\t')
    print('summary shape:', df_summary.shape)
    # ms-ms Data
    print('parse msmsScans')
    infile = os.path.join(TXT_PATH,'msmsScans.txt')
    chunks = find_chunks(infile)
    msmsScans = load_chunks(
        infile=infile,
        usecols = __msms_columns,
        dtype = __dtype,
        tot_chunks=chunks)
    msmsScans.head()
    msmsScans['RT_round']=msmsScans['Retention time'].astype(int)
    msmsScans.loc[:,'RT_bin_qcut'] = msmsScans.groupby('Raw file')['Retention time'].transform(
        lambda x: pd.qcut(x, 4, labels=[1,2,3,4]))


    # msms Identified
    print('parse msmsIdentified')
    msmsIdentified = msmsScans[msmsScans['Identified']=='+'].copy()
    msmsIdentified.loc[:, 'RT_bin_rank'] = msmsIdentified.groupby('Raw file')['Retention time'].rank()
    msmsIdentified.loc[:,'RT_bin_rank_cut'] = msmsIdentified.groupby('Raw file')['RT_bin_rank'].transform(
        lambda x: pd.cut(x, 4, labels=[1,2,3,4], duplicates='drop'))
    msmsIdentified.head()
    # ms Data
    
    df_summary = add_max_retention_time(msmsScans, df_summary)
    df_summary = add_median_injection_time_msms(msmsScans, df_summary)
    df_summary = add_retention_spread(msmsIdentified, df_summary)
    df_summary = add_pep_min(msmsIdentified, df_summary)
    df_summary = add_spray_instability_msms(msmsScans, df_summary)
    
    infile = os.path.join(TXT_PATH,'msScans.txt')
    if os.path.exists(infile):
        print('parse msScans')
        chunks = find_chunks(infile)
        msScans = load_chunks(
            infile=infile,
            usecols = __ms_columns,
            dtype = __dtype,
            tot_chunks=chunks)
        msScans['RT_round']=msScans['Retention time'].astype(int)
        msScans['RT_bin_qcut'] = msScans.groupby('Raw file')['Retention time'].transform(
            lambda x: pd.qcut(x, 4, labels=[1,2,3,4]))
        df_summary = add_median_injection_time_ms(msScans, df_summary)
        df_summary = add_spray_instability_ms(msScans, df_summary)
    else:
        print('msScans does not exist')
        msScans = pd.DataFrame()
        #load the summary file output of maxquant        
        
        
    infile = os.path.join(TXT_PATH,'ms3Scans.txt')
    if os.path.exists(infile):
        print('parse ms3Scans')
        msmsmsScans = load_chunks(
            infile=infile,
            usecols = __msmsms_columns,
            dtype = __dtype,
            tot_chunks=chunks)
        msmsmsScans.head()
        msmsmsScans['RT_round']=msmsmsScans['Retention time'].astype(int)
        msmsmsScans.loc[:,'RT_bin_qcut'] = msmsmsScans.groupby('Raw file')['Retention time'].transform(
            lambda x: pd.qcut(x, 4, labels=[1,2,3,4]))

    else:
        print('ms3Scans does not exist')
        msmsmsScans = pd.DataFrame()
        #load the summary file output of maxquant        

    
    return df_summary, msScans, msmsIdentified, msmsScans, msmsmsScans

#funtion tom make the same plot
def Injection_plot(temp, title, TXT_PATH):
    fig,ax = plt.subplots(figsize=(12,4))
    temp['median_injection_time_msms'].plot(kind='hist', bins=100, ax=ax)
    ax.set_title('Median Ion injection time MSMS')
    ax.set_xlabel('log10 Milliseconds')
    ax.set_ylabel('Count')
    plt.legend([])
    plt.title(title)
    plt.savefig(os.path.join(TXT_PATH,'Injection_Time.png'))
    plt.show()
