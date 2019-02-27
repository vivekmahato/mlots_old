import pandas as pd

from timeseries import TimeSeries

df = pd.read_excel("data.xls", header=[0, 1])
df.columns = df.columns.map('_'.join)

df = df.rename_axis('Lumbar_Gyro X').reset_index()
dic = {'data': df["Lumbar_Gyro X"], 'test': True}
ts = TimeSeries(dic)
del dic
del df
ts.plot()
