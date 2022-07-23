#---Random forest
# Bagging(Breiman 1996) ve Random Subspace (Ho,1998) yöntemlerinin birleşimi ile oluşur
# Ağaçlar için gözlemler bootstrap rastgele örnek seçim yöntemi ile değişkenler random subspace yöntemi ile seçilir.
# Karar ağacının her bir düğümünde en iyi dallara ayırıcı (bilgi kazıyıcı) değişken tüm değişkenler
##arasından RASTGELE seçilen daha az sayıdaki değişken arasından seçilir.
# Ağaçları oluşturmada veri setinin 2/3 kullanılır.Dışarıda kalan veri ağaçların değerlendirilmesi ve değişken öneminin
## belirlenmesi için kullanılır.
# Her düğüm noktasında rastgele değişken seçimi yapılır.(regresyon'da p/3,sınıflamada karekök p)
#Baggingde 100 gözlem olduğunu düşürsek ve bu gözlem sayısına m dersek
##seçilen gözlem sayısı m'den kçük olmak üzere ve n olarak ifade edersek
### burada 75 gözlem gibi düşünelim . Her seferinde 75 ayrı gözlemi değerlendirip
#### yapılan değerlendirmelerin ortalamsını bize getirir.
##### bu yöntem random forest'ın temelini oluşturmaktadır.
#Bagging yönteminde ağaçlar birbirine bağlı değilken
##Boasting yönteminde gözlemler artıklar üzerinden oluşturulur.
#Ezberlemeye karşı dayanıklı bir yöntemdir.


################################################
# Random Forests, GBM, XGBoost, LightGBM, CatBoost
################################################

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("7-Makine_Öğrenmesi/machine_learning/datasets/diabetes.csv")

y = df["Outcome"]
x = df.drop(["Outcome"],axis=1)
################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)

rf_model.get_params()
#parametrelerin deerlerine bakabiliyoruz.
## Önemli olanlar max_depth,criterion,bootstrap,min_samples_split,n_estimators
#n_estimators fit edilecek bağımsız ağaç sayısını ifade etmektedir.
#max_features değişkenlerden kaç tanesini göz önünde bulundurmalıyız. Değişken sayısından fazla olmamalı
#Bunlardaki en iyi değerleri görmek adına Gridsearch iyi optimum değerlerini bulabiliriz.
#İlk olarak şuan ki değerlerimize bakmakta fayda var
cv_results = cross_validate(rf_model,x,y,cv=10,scoring=["accuracy","f1","roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

#öncelikle bu değerleri denemek adına bazı aralıklar vermeliyiz.
rf_params = {"max_depth":[5,8,None],
             "max_features": [3,5,7,"auto"],
             "min_samples_split":[2,5,8,15,20],
             "n_estimators": [100,200,500]}

#Parametre aralıklarını verdikten sonra gridsearch ile en iyi aralığı buluruz

rf_best_grid = GridSearchCV(rf_model,rf_params,cv = 5,n_jobs=1,verbose=True).fit(x,y)
#Fitting 5 folds for each of 180 candidates, totalling 900 fits

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_,random_state=17).fit(x,y)
cv_results = cross_validate(rf_final,x,y,cv=10,scoring=["accuracy","f1","roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()
#Buradan Tek parametreye bakarsak sonuçun değişmediği yönünde olumsuz bir fikire kapılabiliriz.
##Bu yüzden diğer parametrelerede bakmakta fayda var .


#######################
#GBM(Gradient Boosting Machines)
########################

#Adaboost(Adaptive Boosting)
#Zayıf sınıflandırıcıların bir araya gelerek güçlü bir sınıflandırıcı oluşturulması fikrine dayanır.
#2001 Friedman Hatalar/atıklar üzerine tek bir tahminsel model formnda olan moddeller serisi kurulur.
#Gradient boosting tek bir tahminsel model formunda olan modeller serisi oluşturur.
#Seri içersindeki bir model serideki bir önceki modelin tamhmini atıklarını/hatalarının (residuals) üzerine
##kurularak (fit)oluşturur.
#Tek bir tahminsel model formunda olan modeller serisi additive şekilde kurulur.

########### MODEL
gbm_model = GradientBoostingClassifier(random_state=17)

gbm_model.get_params()
#Dikkat etmemiz gereken parametre n_estimators parametresidir.
#Burada optimizasyon anlamına gelmektedir. Random forest'da bağımsız ağaçlara karşılık gelmekteydi

cv_results = cross_validate(gbm_model,x,y,cv=5,scoring=["accuracy","f1","roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

gbm_params = {"max_depth":[3,8,10],
             "learning_rate": [0.01,0.1],
             "subsample":[1,0.5,0.7],
             "n_estimators": [500,1000]}

#subsample parametresi ise tüm gözlemleri mi belirli bir kısmını mı kullanayım diye sorar.
## ön tanımlı değeri 1 dir. Hepsini kullanır.
#Learning_rate öğrenme oranıdır. her ağacın katkısını o oranda küçültür.

gbm_best_params = GridSearchCV(gbm_model,gbm_params,cv=5,n_jobs=1,verbose=True).fit(x,y)

gbm_best_params.best_params_
#Out[20]: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.7}

gbm_final_model = gbm_model.set_params(**gbm_best_params.best_params_,random_state=17,).fit(x,y)

cv_results = cross_validate(gbm_final_model,x,y,cv=5,scoring=["accuracy","f1","roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()


 ################
 #XGBoost
 ###############

 # 2014 XgBoost GBM'in hız ve tahmin performansını arttırmak üzere optimize edilmiş ;
 ## ölçeklenebilir ve farklı platfromlara entegre edilebilir versiyonudur.


xgboost_model = XGBClassifier(random_state=17)

cv_results = cross_validate(xgboost_model,x,y,cv=5,scoring=["accuracy","f1","roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

xgboost_params = {"max_depth":[5,8,None ],
             "learning_rate": [0.01,0.1],
             "colsample_bytree":[None,0.7,1],
             "n_estimators": [100,500,1000]}

xgboost_best_grid = GridSearchCV(xgboost_model,xgboost_params,cv=5,n_jobs=-1,verbose=True).fit(x,y)

xgboost_best_grid.best_params_

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_,random_state=17).fit(x,y)

cv_results = cross_validate(xgboost_final,x,y,cv=5,scoring=["accuracy","f1","roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

###################################
# LGBM (LightGBM)
##################################

#LightGBM , XGboost'un eğitim süresi perfonmansını arttırmaya yönelik geliştirilen diğer bir GBM'türüdür.
#Level-wise büyüme stratejisi yerine leaf wise büyüme stratejisi ile daha hızlıdır.


################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Hiperparametre yeni değerlerle
lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [200, 300, 350, 400],
               "colsample_bytree": [0.9, 0.8, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


# Hiperparametre optimizasyonu sadece n_estimators için.
lgbm_model = LGBMClassifier(random_state=17, colsample_bytree=0.9, learning_rate=0.01)

lgbm_params = {"n_estimators": [200, 400, 1000, 5000, 8000, 9000, 10000]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=True)
#Verbose False yapmazsanız uzun bir çıktı ile karşılaşacaksınız


cv_results = cross_validate(catboost_model, x, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

################################
# Hyperparameter Optimization with RandomSearchCV (BONUS)
################################

rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(X, y)


rf_random.best_params_


rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_random_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################
# Analyzing Model Complexity with Learning Curves (BONUS)
################################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]


rf_model = RandomForestClassifier(random_state=17)

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

rf_val_params[0][1]



