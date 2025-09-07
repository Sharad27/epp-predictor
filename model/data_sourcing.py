import pandas as pd

class Data_Sourcing:
    '''
    Creating the EPP Taker Class
    '''
    def __init__(self,df:pd.DataFrame=None):
        if df is not None and not isinstance(df,pd.DataFrame):
            raise TypeError(f'Expected a pandas dataframe as input but the input type passed is {type(df)}')
        self.df=df

    def replace_nulls(self):
        '''
        Function to impute missing values conditionally
        '''
        import pandas as pd

        null_col_list=[]
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if self.df[col].dtype in ['int64','float64']:
                    # Imputing missing numeric values with median if distribution is skewed, mean if it is a normal distribution (i.e. df.skew()=0):
                    self.df[col]=self.df[col].fillna(self.df[col].mean() if abs(self.df[col].skew())<=0.5 else self.df[col].median())
                else:
                    if self.df[col].mode().empty:
                        self.df[col]=self.df[col].fillna('')
                    else:
                        self.df[col]=self.df[col].fillna(self.df[col].mode()[0])

        return self.df
    
    def create_utilization(self):
        '''
        Function to create utilization column
        '''
        import numpy as np
        self.df['UTILIZATION'] = self.df.apply(lambda row: row['BALANCE']/row['CREDIT_LIMIT'] if row['CREDIT_LIMIT'] != 0 else 0, axis=1)
        return self.df

    def create_label(self):
        '''
        Function to generate label column
        '''
        import numpy as np
        # Synthetically creating a binary label, since this dataset is clustering based:
        conditions = (
            (self.df['PURCHASES'] > self.df['PURCHASES'].median()) &   # high spender
            (self.df['INSTALLMENTS_PURCHASES'] > self.df['ONEOFF_PURCHASES']) &  # prefers installments
            (self.df['UTILIZATION'] > 0.5) &                      # high credit utilization
            (self.df['PRC_FULL_PAYMENT'] < 0.5) &                   # not a full payer
            (self.df['PAYMENTS'] < 0.6 * (self.df['PURCHASES']+1))     # pays less than 60% of purchases
            )

        self.df['EPP_TAKER'] = np.where(conditions, 1, 0)
        return self.df

    def handling_imbalance(self,threshold=0.4):
        '''
        Function to conditionally handle class imbalance using SMOTE
        '''
        from imblearn.over_sampling import SMOTE

        majority_class_obs=max(len(self.df.query('EPP_TAKER==0')),len(self.df.query('EPP_TAKER==1')))
        minority_class_obs=len(self.df)-majority_class_obs

        ratio=minority_class_obs/majority_class_obs

        if ratio<threshold:
            smote=SMOTE()
            X=self.df.drop('EPP_TAKER',axis=1)
            y=self.df['EPP_TAKER']
            X_res,y_res=smote.fit_resample(X,y)
            self.df_resampled=pd.DataFrame(pd.concat([X_res,y_res],axis=1),columns=self.df.columns)
            return self.df_resampled
        else:
            return self.df
           
    def feature_selection(self,X_train,correlation_threshold=0.9):
        '''
        Function to remove highly correlated features
        '''
        import numpy as np

        corr_matrix=X_train.corr().abs()
        upper_tri=corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
        to_drop=[column for column in upper_tri.columns if any(upper_tri[column]>correlation_threshold)]
        X_train_reduced=X_train.drop(columns=to_drop,axis=1)
        # manually select features
        X_train_reduced=X_train_reduced[['CREDIT_LIMIT','BALANCE','UTILIZATION','PURCHASES','CASH_ADVANCE','PAYMENTS','TENURE']]
        return X_train_reduced
          
    def scale_features(self,X_train,X_test):
        '''
        Function to scale the features using StandardScaler
        '''
        from sklearn.preprocessing import StandardScaler
        import joblib
        import os

        scaler=StandardScaler()
        X_train_scaled=scaler.fit_transform(X_train)
        X_test_scaled=scaler.transform(X_test)
        # dump the scaler
        joblib.dump(scaler,os.path.join(os.path.dirname(os.path.abspath(__file__)),"scaler.pkl"))

        return X_train_scaled,X_test_scaled

    def export_feature_names(self,data: pd.DataFrame):
        '''
        Function exporting the final feature names as a pickle file
        '''
        import joblib
        import os

        if data is not None and not isinstance(data,pd.DataFrame):
            raise TypeError(f'Expected a pandas dataframe as input but the input type passed is {type(df)}')
        # dump the final feature names
        joblib.dump(data.columns,os.path.join(os.path.dirname(os.path.abspath(__file__)),"feature_names.pkl"))

    def format_probability(self,prob):
        '''
        Function to conditionally display decimal points
        '''
        pct = prob * 100
        if pct >= 1:  
            # For >= 1%, show 2 decimals (cleaner)
            return f"{pct:.2f}%"
        elif pct >= 0.01:  
            # For 0.01% – 1%, show 4 decimals
            return f"{pct:.4f}%"
        else:  
            # For very tiny probabilities, use scientific notation
            return f"{pct:.2e}%"