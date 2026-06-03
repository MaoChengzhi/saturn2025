import pandas as pd
import numpy as np
import pickle





def select_df(start_time,end_time):
    
    '''
    select files that cover the time range from start_time to end_time
    input: start_time, end_time - string or pd.Timestamp
    output: selected_files - pd.DataFrame with columns file_name, start_time, end_time
    '''
            
            
    with open('data/sec-krtp-file_time_index.csv','rb') as f:
        index_df=pickle.load(f)
    
    start_time=pd.to_datetime(start_time)
    end_time=pd.to_datetime(end_time)
    
    # select files that cover the time range
    index_mask=(index_df['end_time']>=start_time) & (index_df['start_time']<=end_time)
    selected_index_df=index_df[index_mask]
    
    df=pd.DataFrame()
    for file in selected_index_df['file_name']:
        temp_df=pd.read_csv(file,sep=r'\s+',names=['TIME','BR','BTHETA','BPHI','BTOTAL','NPTS'])
        df=pd.concat([df,temp_df],ignore_index=True)
        
    df['TIME']=pd.to_datetime(df['TIME'])
    mask=(df['TIME']>=start_time) & (df['TIME']<=end_time)
    df=df.loc[mask].reset_index(drop=True)
    
    return df


def detrend_data(df):
    '''detrend the data by removing the linear trend'''
    df.set_index('TIME',inplace=True)
    df=df-df.rolling(window=1200,center=True,
                    #  min_periods=1,
                     ).mean()   # 20 minutes window
    df.reset_index(inplace=True)
    
    # keep the rows where 'BR' is not nan, i.e., truncate the starting and ending nan values
    df=df[df['BR'].notna()].reset_index(drop=True)
    
    return df

def fac_decompose_dB(df,start_time,end_time):
    
    # start_time=pd.to_datetime('2010-12-21 02:50:00')
    # end_time=pd.to_datetime('2010-12-21 04:20:00')


    B=df[['BR','BTHETA','BPHI']].to_numpy()
    B0=df[['BR','BTHETA','BPHI']].rolling(window=1200,center=True,).mean()
    # Roll the DataFrame/Series, not the NumPy array B. Reason: rolling() is a pandas feature that understands your time index, handles center=True, missing data, alignment, and it writes back cleanly into df. If you roll a plain NumPy array, you’ll have to manually handle edge effects, NaNs, and alignment.
    B0=B0.to_numpy() # or rename(columns={'BR':'B0_R','BTHETA':'B0_THETA','BPHI':'B0_PHI'})
    df[['B0_R','B0_THETA','B0_PHI']]=B0

    df['B0_TOTAL']=np.linalg.norm(B0,axis=1)

    dB=B - B0

    df[['dB_R','dB_THETA','dB_PHI']]=dB
    df['dB_TOTAL']=np.linalg.norm(dB,axis=1)

    b_hat=B0 / np.linalg.norm(B0,axis=1)[:,np.newaxis]

    b_hat=B0 / np.linalg.norm(B0,axis=1)[:,np.newaxis]
    df[['b_hat_R','b_hat_THETA','b_hat_PHI']]=b_hat

    dB_parallel=np.sum(dB * b_hat,axis=1) # einsum('ij,ij->i',dB,b_hat)
    df['dB_parallel']=dB_parallel

    df['dB_perpendicular']=np.sqrt(df['dB_TOTAL']**2 - df['dB_parallel']**2)

    r_hat=np.zeros(b_hat.shape)
    r_hat[:,0]=1  # r_hat = (1,0,0)
    # Gram-Schmidt process to get the perpendicular components
    # Gram–Schmidt: v = r - (r·b) b
    #  r=(1,0,0), r·b = b_R

    e_perp_1= (r_hat - np.sum(r_hat * b_hat, axis=1).reshape(-1,1) * b_hat  )
    # e_perp_1= (r_hat - np.einsum('i,ij->ij', (r_hat * b_hat).sum(axis=1), b_hat)  )
    # e_perp_1= (r_hat - (r_hat * b_hat).sum(axis=1)[:,np.newaxis] * b_hat)

    e_perp_1= e_perp_1 / np.linalg.norm(e_perp_1,axis=1).reshape(-1,1)  # normalize

    e_perp_2= np.cross(b_hat,e_perp_1) #,axis=1 or -1 )  # cross product to get the second perpendicular unit vector

    dB_perpendicular_1=np.sum(dB * e_perp_1,axis=1)
    dB_perpendicular_2=np.sum(dB * e_perp_2,axis=1)
    df['dB_perpendicular_1']=dB_perpendicular_1
    df['dB_perpendicular_2']=dB_perpendicular_2

    # delete the first and last 10 minutes to avoid nan values
    df=df[(df['TIME']>=start_time+pd.Timedelta(minutes=10)) & (df['TIME']<=end_time-pd.Timedelta(minutes=10))].reset_index(drop=True)
    return df

def read_and_process_data(start_time,end_time):
    df=select_df(start_time,end_time)
    # df=detrend_data(df)
    df=fac_decompose_dB(df,start_time,end_time)
    return df