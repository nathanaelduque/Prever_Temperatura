import pandas as pd
import numpy as np 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KNeighborsRegressor
import mlflow 
from mlflow.pyfunc import PythonModel
from mlflow.utils.environment import _mlflow_conda_env
from sklearn.svm import SVR
from prophet import Prophet



#Importa os dados do github
URL = "jena_climate_2009_2016.csv"
dados=pd.read_csv(URL, index_col=['Date Time'])

#Calculando para dados de hora em hora 
dados.index= pd.to_datetime(dados.index,dayfirst = True)
dados_hora = dados.resample('1H').mean()


#Retirando o Tpot (K)
dados_hora = dados_hora.drop('Tpot (K)',axis = 1)

#Dando shift nos dados 
dados_hora['y_shifted'] = dados_hora['T (degC)'].shift(periods = -2)
dados_hora = dados_hora[:-2]

#Apagando os dados criados pelo comando dados_hora = dados.resample('1H').mean() 
#devido a falta de dados
dados_hora = dados_hora.dropna()

##Criando a coluna mês
dados_hora['mes'] = dados_hora.index.month

#Calculando os valores das baselines e a Baseline 
baseline_verao =  dados_hora.query("mes in [7,8,9]")['T (degC)'].mean()
baseline_inverno = dados_hora.query("mes in [1,2,3]")['T (degC)'].mean()
baseline_transicao = dados_hora.query("mes in [4,5,6,10,11,12]")['T (degC)'].mean()

dados_hora['baseline'] = np.where(dados_hora['mes'].isin([7,8,9]), baseline_verao,np.where(dados_hora['mes'].isin([1,2,3]),baseline_inverno,baseline_transicao))

#Calculando os parâmetros da Baseline 
mse_base = mean_squared_error(dados_hora['T (degC)'], dados_hora.baseline)
mae_base = mean_absolute_error(dados_hora['T (degC)'], dados_hora.baseline)
rmse_base = sqrt(mse_base)

dados_hora = dados_hora.reset_index()
#Inserindo as variáveis logarítimicas no dataset
dados_hora['log_VPmax (mbar)'] =  np.log(dados_hora['VPmax (mbar)'])
dados_hora['log_VPact (mbar)'] =  np.log(dados_hora['VPact (mbar)'])
dados_hora['log_VPdef (mbar)'] =  np.log(dados_hora['VPdef (mbar)'])
dados_hora['log_sh (g/kg)'] =  np.log(dados_hora['sh (g/kg)'])
dados_hora['log_H2OC (mmol/mol)'] =  np.log(dados_hora['H2OC (mmol/mol)'])


#Dropando dados desnecessários
dados_final =  dados_hora.drop(['p (mbar)','rh (%)','VPmax (mbar)','VPact (mbar)','sh (g/kg)','H2OC (mmol/mol)'
                                ,'wv (m/s)','max. wv (m/s)','wd (deg)','log_VPdef (mbar)','log_VPdef (mbar)','mes','baseline','VPdef (mbar)'],axis = 1)




#Dividindo o dataset en treino e teste e normalizando-o 
x  = dados_final.drop(['y_shifted','Date Time'], axis = 1)


pt = PowerTransformer()
pt.fit(x)

x = pt.transform(x)

y = dados_final['y_shifted']

treino_x  = x[:int(0.75*len(x))]

teste_x  = x[int(0.75*len(x)):]

treino_y =  y[:int(0.75*len(y))]

teste_y  = y[int(0.75*len(y)):]

#Setando nosso experimento, criando nosso modelo de KNN e passando os parâmetros para o mlflow 
mlflow.set_experiment('FIEC')
mlflow.start_run()





#Criando o Modelo do Prophet e passando para o MLFlow 



#Fazendo ajustes no dataframe para ele se adequar ao formato dessa biblioteca 

dados_prophet = pd.DataFrame(pt.transform(dados_final.drop(['Date Time','y_shifted'],axis = 1)))
dados_prophet = dados_prophet.rename(columns = {0:'T (degC)' ,1:'Tdew (degC)' ,2:'rho (g/m**3)' ,3:'log_VPmax (mbar)',4: 'log_VPact (mbar)',
                                               5:'log_sh (g/kg)',6: 'log_H2OC (mmol/mol)'})
dados_prophet['ds'] =  dados_final['Date Time']
dados_prophet['y'] = dados_final['y_shifted']

# inicializa o modelo
modelo_prophet = Prophet(interval_width = 0.95)
# adiciona as fetures escolhidas para ajudar na previsão
modelo_prophet.add_regressor('T (degC)')

modelo_prophet.add_regressor('Tdew (degC)')

modelo_prophet.add_regressor('rho (g/m**3)')

modelo_prophet.add_regressor('log_VPmax (mbar)')

modelo_prophet.add_regressor('log_VPact (mbar)')

modelo_prophet.add_regressor('log_sh (g/kg)')

modelo_prophet.add_regressor('log_H2OC (mmol/mol)')



# fit do modelo no treino
modelo_prophet.fit(dados_prophet.iloc[:int(0.75*len(dados_prophet))])

mlflow.prophet.log_model(modelo_prophet, 'prophet')

# previsao nos dados de teste
previsao_prophet = modelo_prophet.predict(dados_prophet.iloc[int(0.75*len(dados_prophet)):])

mse_prophet = mean_squared_error(dados_prophet.iloc[int(0.75*len(dados_prophet)):]['y'], previsao_prophet['yhat'])
mae_prophet = mean_absolute_error(dados_prophet.iloc[int(0.75*len(dados_prophet)):]['y'], previsao_prophet['yhat'])
rmse_prophet = sqrt(mse_prophet)

mlflow.log_metric('mse', mse_prophet)
mlflow.log_metric('rmse', rmse_prophet)
mlflow.log_metric('mae', mae_prophet)


print("Tracking URI: ", mlflow.get_tracking_uri())
print("Essa:",mlflow.get_artifact_uri())




conda_env = _mlflow_conda_env(
    additional_conda_deps=[
        "keras=={}".format(keras.__version__),
        "tensorflow=={}".format(tf.__version__),
    ],
    additional_pip_deps=[
        "cloudpickle=={}".format(cloudpickle.__version__),
    ])

class KerasMnistCNN(PythonModel):

    def load_context(self, context):
        import tensorflow as tf
        self.graph = tf.Graph()
        with self.graph.as_default():
            K.set_learning_phase(0)
            self.model = mlflow.keras.load_model(context.artifacts["keras-model"])

    def predict(self, context, input_df):
        with self.graph.as_default():
            return self.model.predict(input_df.values.reshape(-1, 28, 28))

mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=KerasMnistCNN(),
    artifacts={
        "keras-model": mlflow.get_artifact_uri("model")
    },
    conda_env=conda_env)

mlflow.end_run()