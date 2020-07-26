
import os
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
%matplotlib auto

# print(os.path.dirname(__file__))

currentDirectory = os.path.dirname(__file__)
DataDirectory = os.path.join(currentDirectory,'Data')

# os.chdir(currentDirectory)
# print(os.listdir(currentDirectory))
# print(os.listdir(DataDirectory))
print("================= Reading Data ===================")
DataDict = {}
for file in os.listdir(DataDirectory):
    FileName = file.split('_')[0]
    DataPath = os.path.join(DataDirectory,file)
    try:
        Data = pd.read_csv(DataPath,parse_dates=['Datetime'])
    except Exception as e:
        print(file,e)
        Data = pd.read_csv(DataPath)
    DataDict[FileName] = Data
    
Train = DataDict['Train'].copy() 
Test = DataDict['Test'].copy() 


print("=================Data Exploration===================")
PlotPath = os.path.join(currentDirectory,"DataExploration")

Train.iloc[:,[1,2]].set_index('Datetime').plot()
plt.title('2012 to 2013 Counts')
plotname = 'TrainPlot.png'
# asdasd
plt.savefig(os.path.join(PlotPath,plotname))

print("=================DataPreprocessing===================")
# TrainPlot = Train.copy()
#Resmoving Multiplicative Seannality
Train['Count'] =  np.log( Train['Count'])
# Remving Trend
# Train['Count'] = Train['Count'] - Train['Count'].shift( )
# Train['Count'] = Train['Count'].diff()

# Train['Count'].rolling(2).sum().shift(-1)


Train.iloc[:,[1,2]].set_index('Datetime').plot()
plt.title('2012 to 2013 Counts After Preprocessing')
plotname = 'PreprocessedTrainPlot.png'
# asdasd
plt.savefig(os.path.join(PlotPath,plotname))


def preprocess(Data):
      data = Data.copy()
      # data.drop('ID',axis=1,inplace=True)
      data['year'] = data['Datetime'].dt.year
      data['month'] = data['Datetime'].dt.month
      data['day'] = data['Datetime'].dt.day
      data['hour'] = data['Datetime'].dt.hour
      return data



Train = preprocess(Train)
Test = preprocess(Test)

exogenous_features =['year', 'month', 'day', 'hour']
print("=================ModelBuilding===================")
 # Model Building

model = Prophet()
model.fit(Train[['Datetime','Count']+exogenous_features].rename(columns = {'Datetime':'ds','Count':'y'}))

ResultsPath = os.path.join(currentDirectory,"Results")
forecast = model.predict(Test[['Datetime']+exogenous_features].rename(columns={'Datetime':'ds'}))
PLtName = 'ModelComponents.png'


# future = model.make_future_dataframe(periods=213)
# forecast = model.predict(future)
# model.plot(forecast);

model.plot_components(forecast);
# model.plot(forecast);
plt.savefig(os.path.join(ResultsPath,PLtName))

Output = pd.DataFrame()
Output['ID'] = DataDict['sample']['ID'].copy()
Output['Count'] = forecast['yhat']
# CountPreds = pd.Series(Train_log.ix[0])
# Output['Count'] = Output['Count'].cumsum().shift(-1)
# Output['Count'].fillna(method = 'ffill',inplace=True)
# Output['Count']  =  # code for int 
Output['Count']  = np.exp( Output['Count']) 

Output.to_csv(os.path.join(ResultsPath,'PropFeatureFinal.csv'),index=False)




# model.fit(PrepTrain[['DateTime','Vehicles']+exogenous_features].rename(columns={'DateTime':'ds','Vehicles':'y'}))
















