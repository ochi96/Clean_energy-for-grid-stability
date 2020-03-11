import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

#normalising the input data using the test data
def normalize(data):
        sc = StandardScaler()
        data_numerical_features = list(data.select_dtypes(include=['int64','float64','int32','float32']).columns)
        data_scaled = pd.DataFrame(data)
        data_scaled[data_numerical_features] = sc.fit_transform(data_scaled[data_numerical_features])
        new = data_scaled.iloc[:1]
        return new

#predictions
def testlist(data):
        loaded_model = pickle.load(open("model.pkl",'rb'))
        result = loaded_model.predict(data).flatten()    
        return result[0]

#datareceived = pd.DataFrame({'AT':19.07,'V':49.69,'AP':1007.22,'RH':76.79},index=[0])
#test_df = pd.read_csv('power_test.csv')
#test_data = test_df.drop(['PE'],axis=1)
#test_data = pd.concat([datareceived, test_data]).reset_index(drop = True)
#normalized = normalize(test_data)
#answer = testlist(normalized)

