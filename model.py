import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
df=pd.read_csv('MobileTrain.csv')
X=df[['ram','battery_power','px_height','px_width']]
y=df['price_range']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.20)
r =LogisticRegression(multi_class='multinomial', solver='lbfgs')
m=r.fit(X_train,y_train)
pickle.dump(r,open('model.pkl','wb') )
