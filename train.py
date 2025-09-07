import os
import pandas as pd
from config import DATA_DIR
from model.model_training import Model_Training
from model.data_sourcing import Data_Sourcing

import warnings
warnings.filterwarnings(action='ignore')

if __name__ == "__main__":

    # Load data
    df=pd.read_csv(os.path.join(DATA_DIR,'input_data.csv')).drop('CUST_ID',axis=1)

    # Initialize trainer with df
    trainer=Model_Training(df)
    # Preprocessing 
    df=trainer.replace_nulls() # replacing null column values
    df=trainer.create_utilization() # creating utilization column
    df=trainer.create_label() # creating label column for converting clustering to classification problem
    df=trainer.handling_imbalance(threshold=0.4) # conditionally applying SMOTE if imbalance ratio is below threshold
    # Train model
    X_train,X_test,y_train,y_test=trainer.split_data(data=df,test_size=0.2,stratify=True,random_state=42)
    X_train_reduced=trainer.feature_selection(X_train,correlation_threshold=0.8) # removing highly correlated features on the basis of threshold

    X_test_reduced=X_test[X_train_reduced.columns] # matching column names b/w X_train and X_test
    X_train_scaled,X_test_scaled=trainer.scale_features(X_train=X_train_reduced,X_test=X_test_reduced) # feature scaling    

    trainer.train_model(X_train=X_train_scaled,X_test=X_test_scaled,y_train=y_train,y_test=y_test)
    trainer.export_feature_names(data=X_train_reduced)
