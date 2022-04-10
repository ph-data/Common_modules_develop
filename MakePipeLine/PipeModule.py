import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

import cloudpickle
import joblib


class PipeProcess:
    def __init__(self,steps, model, param=False, cv=5, remainder='drop'):
        """
        Parameters
        ----------
        ### 입력변수별 적용함수 및 파이프명
        steps : dict
        steps={
                '파이프명':[[(함수명,적용함수),..],[적용변수,..]],
                '파이프명2':[[(함수명2,적용함수2),..],[적용변수2,..]]
                ...
                  }
        ex)
        steps={
            'num_pipe': [[('imputer',SimpleImputer(fill_value=np.nan)),('std_scaler', StandardScaler())],['Age', 'Fare']],
            'cat_pipe': [[('imputer',SimpleImputer(fill_value=np.nan, strategy='most_frequent'))
                          ,('cat_encoder', OneHotEncoder(handle_unknown='ignore'))],[ 'Sex', 'SibSp' , 'Cabin', 'Embarked']]
                }


        ### 모델 
        model : model
        model = 모델()
        ex) model = RandomForestClassifier(random_state=100)

        
        ### GridSearchCV의 사용될 param_grid 매개변수, 매개변수 적용을 위한 단계별 이름을'__'으로 연결하여야 함
        param : dict
        param = {
                'preprocess__파이프명__함수명__함수매개변수':[적용범위] #전처리 단계의 매개변수
                'model__모델매개변수':[적용범위]  #모델 단계의 매개변수
                }
        ex)
        param = {
           'preprocess__num_pipe__imputer__strategy': ["mean", "median", "most_frequent"],
           'model__n_estimators' : [10, 100],  
           'model__max_depth' : [6, 8, 10, 12]
                }

        ### 교차검증
        cv : int or KFold
        cv=교차검증 수 또는 KFold(),default=5
        
        ex)
        cv=10 
        cv=KFold(n_splits=3, shuffle=True, random_state=1)
        
        ###sklearn.compose.ColumnTransformer의 remainder 매개변수 
        remainder : str
        remainder= 'drop' or 'passthrough' default='drop'
        ex)
        remainder='drop' #기본적으로 의 지정된 열만 transformers변환되어 출력에서 결합되고 지정되지 않은 열은 삭제
        remainder='passthrough'#지정 되지 않은 나머지 모든 열이 transformers자동으로 전달
        
        """
        
        self.steps=steps 
        self.model=model
        self.param=param
        self.cv=cv
        self.remainder=remainder

    def preprocess_pipe(self, steps):
        """
        전처리단계의 파이프라인을 구성 
        
        """
        pipe_list=[]
       
        for i,j in steps.items():
            pipe_x=Pipeline(j[0])
            pipe_list.append((i,pipe_x,j[1]))
        
        preprocessor=ColumnTransformer(pipe_list, remainder=self.remainder)
        
        return preprocessor
    
    def model_pipe(self,model,preprocessor):
        """
        전처리 파이프라인과 모델을 연결하여 최종 파이프라인 구성
        
        """
        pipe = Pipeline([
            ('preprocess', preprocessor),
            ('model', model)
        ])
        return pipe
    
    def gridsearch_pipe(self, pipe, param, cv):
        """
        파이프라인 그리드서치 수행
        
        """
        gs_model=GridSearchCV(pipe, param_grid=param, cv=cv)
        return gs_model
    
    @staticmethod
    def save_pipe(pipe_name,file_path):
        """
        최종 파이프라인 Save
        
        Parameters
        ----------
        pipe_name = 저장할 파이프라인
        
        file_path='파일경로'
        
        """
        cloud_var=cloudpickle.dumps(pipe_name)
        joblib.dump(cloud_var, "{}.pkl".format(file_path))
        return print("{}.pkl".format(file_path))
    
    @staticmethod
    def load_pipe(file_path):
        """
        저장한 파이프라인 Load
        객체생성 없이 PipeModule.load_pipe(file_path) 형태로 접근하여 사용 가능
        
        Parameters
        ----------
        file_path='파일경로'
        
        """
        file_var=joblib.load("{}.pkl".format(file_path))
        pipe_var=cloudpickle.loads(file_var)
        return pipe_var
    
    def get_pipe(self):
        """
        전체 파이프라인 생성 과정 수행
        
        Parameters
        ----------
        param 매개변수를 입력하지 않을 시 GridSearchCV를 수행하지 않은 모델파이프라인 return
        
        """
        pr1=self.preprocess_pipe(self.steps)
        pr_model=self.model_pipe(self.model,pr1)
        
        if not self.param:
            pipe_name=pr_model
    
        else:
            pipe_name=self.gridsearch_pipe(pr_model,self.param,self.cv)
        
        return pipe_name
