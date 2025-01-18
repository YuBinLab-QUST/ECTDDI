import numpy as np
import pandas as pd
#from imblearn.ensemble import EasyEnsemble
import joblib
from sklearn.preprocessing import scale,StandardScaler
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN,SMOTETomek

data_=pd.read_csv('ET50.csv')
data1=np.array(data_)
data=data1[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((47947,1))#Value can be changed
label2=np.zeros((100737,1))
label=np.append(label1,label2)
shu=scale(data)

X=shu
y=label#.astype('int64')

# =============================================================================
# rus = RandomUnderSampler(random_state=0)
# X_resampled, y_resampled = rus.fit_sample(X, y)
# =============================================================================

# =============================================================================
# iht = InstanceHardnessThreshold(estimator=LogisticRegression())
# X_resampled, y_resampled = iht.fit_sample(X, y)
# =============================================================================
# =============================================================================
#nm1 = NearMiss(version=1)
#X_resampled, y_resampled = nm1.fit_sample(X, y)
# =============================================================================
# =============================================================================
#cc = ClusterCentroids(random_state=0)
#X_resampled, y_resampled = cc.fit_sample(X, y)
# =============================================================================
#print sorted(counter(y_resampled).items())
# =============================================================================
#tl = TomekLinks()
#X_resampled, y_resampled = tl.fit_sample(X, y)
# =============================================================================
# X_resampled_adasyn, y_resampled_adasyn = ADASYN(sampling_strategy={1:17538}).fit_sample(X, y)
# X_resampled_, y_resampled_ = RandomUnderSampler(sampling_strategy={0:17538}).fit_sample(X_resampled_adasyn, y_resampled_adasyn)
#oss = OneSidedSelection(random_state=0)
#X_resampled, y_resampled = oss.fit_sample(X, y)

# =============================================================================
# enn = EditedNearestNeighbours()
# X_resampled, y_resampled = enn.fit_sample(X, y)
# =============================================================================
# =============================================================================
# renn = RepeatedEditedNearestNeighbours()
# X_resampled, y_resampled = renn.fit_sample(X, y)
# =============================================================================
#sorted(Counter(y_resampled_adasyn).items())
#X_resampled, y_resampled = nm1.fit_sample(X, y)
#smote_enn = SMOTEENN()
#X_resampled, y_resampled = smote_enn.fit_sample(X, y)
#smote_enn = SMOTEENN(random_state=0)
#X_resampled, y_resampled = smote_enn.fit_resample(X, y)
# =============================================================================
#X_resampled, y_resampled = SMOTETomek(sampling_strategy={1:60000}).fit_resample(X, y)
#X_resampled_, y_resampled_ = RandomUnderSampler(sampling_strategy={0:50646}).fit_resample(X_resampled, y_resampled)
# =============================================================================
#X_resampled, y_resampled = SMOTEENN(sampling_strategy={1:81000}).fit_resample(X, y)
#X_resampled_, y_resampled_ = RandomUnderSampler(sampling_strategy={0:47000}).fit_resample(X_resampled, y_resampled)
# =============================================================================
X_resampled, y_resampled = SMOTE(sampling_strategy={1:160000}).fit_resample(X, y)
X_resampled_, y_resampled_ = RepeatedEditedNearestNeighbours().fit_resample(X_resampled, y_resampled)
# =============================================================================

shu2 =X_resampled_
shu3 =y_resampled_
data_csv = pd.DataFrame(data=shu2)
data_csv.to_csv('SMOTE-RENN_50.csv')
data_csv = pd.DataFrame(data=shu3)
data_csv.to_csv('SMOTE-RENN_label_50.csv')