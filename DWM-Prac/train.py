import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Housing.csv")
dataset.drop(columns={"stories", "hotwaterheating", "prefarea", "bathrooms","airconditioning"}, inplace=True)
map_mainroad = {"yes":1, "no":0}
map_guestroom = {"yes": 1, "no":0}
map_basement = {"yes": 1, "no":0}
map_furnishingstatus = {'furnished':1, 'semi-furnished':2, 'unfurnished':0}
dataset.mainroad = dataset.mainroad.map(map_mainroad)
dataset.guestroom = dataset.guestroom.map(map_guestroom)
dataset.basement = dataset.basement.map(map_basement)
dataset.furnishingstatus = dataset.furnishingstatus.map(map_furnishingstatus)
y = dataset.price
x = dataset.drop(columns={'price'})

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
models = [RandomForestRegressor(n_estimators=10), LinearRegression(), KNeighborsRegressor(n_neighbors=10)]
for model in models:
  predictor = model.fit(x_train, y_train)
  y_pred = predictor.predict(x_test)
  ac = mean_absolute_error(y_test, y_pred)
  ac1 = r2_score(y_test, y_pred)
  print(f"For {model}")
  print(f"MSE score is {ac}")
  print(f"r2_score is {ac1}")
  print("------------------------------")
model = RandomForestRegressor(n_estimators=10)
model.fit(x_train, y_train)
pickle.dump(model, open("trained.pkl", "wb"))