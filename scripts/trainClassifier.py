import cv2
import numpy as np;
import pandas as pd
import matplotlib.pyplot as plt
cv2.useOptimized()
from sklearn import preprocessing

from searchwing import roiDescriptor

#path to the hdf5 file with the descriptors
descriptorsPathIn="/extracedDescriptors.h5"
#filepath to output the learned classifer as pickle file
classifierPathOut="/classifier.pkl"

print("\nRead dataset with descriptors from "+descriptorsPathIn)
df_raw = pd.read_hdf(descriptorsPathIn,
                key="table")
df_raw=df_raw.dropna(axis=0)
df_raw.shape
print("Dataset shape:"+str(df_raw.shape)+" (Entrys,Features)")
print("Dataset got the following classdistribution")
print(pd.value_counts(df_raw["class"]))

#Weight nature samples by pixcount to get more samples with high pixcount for training
df_boat=df_raw[df_raw["class"]=="boat"]
sampleCountNature=int(df_boat.shape[0])

df_nature = df_raw[df_raw["class"]=="nature"]
weights = 5*df_nature["pixcount"]
weightsnp=np.array(weights.values,dtype=float)
weightsnp=weightsnp/sum(weightsnp) #norm
indizes=range(0,len(weightsnp))
sampleIndizes=np.random.choice(indizes,sampleCountNature,replace=False, p=weightsnp)
df_nature = df_nature.iloc[sampleIndizes,:] #df_nature.sample(n=654,weights="pixcount",axis=0)


df=pd.concat([df_nature,df_boat])
print("Sample same amount of boats...")
print("Dataset got the following classdistribution")
print(pd.value_counts(df["class"]))
Y=df['class']
X=df

from sklearn.model_selection import train_test_split
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.25)

X_train=X_train.iloc[:,9:]  #-11 skip hmoments
X_test_meta=X_test.iloc[:,0:8]
X_test=X_test.iloc[:,9:]#

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
print("\nLearn RandomForest classifier...")

clf = RandomForestClassifier(n_estimators=25,criterion="entropy",n_jobs=4,class_weight="balanced")
clf = clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = pd.DataFrame(clf.predict(X_test) )

# Compute confusion matrix
print("Compute confusion matrix")
cnf_matrix_train = confusion_matrix(y_train, y_pred_train)
np.set_printoptions(precision=2)
print(cnf_matrix_train)
cnf_matrix_test = confusion_matrix(y_test, y_pred_test)
np.set_printoptions(precision=2)
print(cnf_matrix_test)
print("Accuracy:",accuracy_score(y_test,y_pred_test))

from sklearn.externals import joblib
print("Save classifier to " + classifierPathOut +" ...")
joblib.dump(clf, classifierPathOut) 

#Print feature importances of learned classifier
print("\nPrint feature importances of learned classifier")
imp=clf.feature_importances_

descr= roiDescriptor.Descriptors()
colList=list(X_train.columns.values)

colorChannels=descr.channelNames
colorChannelsImportances = np.zeros([len(colorChannels)])
featureTypes=["mean","std","entropy","Hist","LBP","Mask"]
featureTypesImportances = np.zeros([len(featureTypes)])

for colIdx,oneColName in enumerate(colList,0):
    for chanIdx,oneChanName in enumerate(colorChannels,0):
        if oneChanName in oneColName:
            colorChannelsImportances[chanIdx]+=imp[colIdx]
    for featTypeIdx,oneFeatTypeName in enumerate(featureTypes,0):
        if oneFeatTypeName in oneColName:
            featureTypesImportances[featTypeIdx]+=imp[colIdx]

print("FeatureImportances for each Channel:")
for chanIdx,oneChanName in enumerate(colorChannels,0):
    print(oneChanName,colorChannelsImportances[chanIdx])
#print("Sum",sum(colorChannelsImportances))
print("FeatureImportances for each Featuretype:")
for featTypeIdx,oneFeatTypeName in enumerate(featureTypes,0):
    print(oneFeatTypeName,featureTypesImportances[featTypeIdx])

print("Show feature importances for each feature:")
imp=clf.feature_importances_
for i in xrange(len(X_train.columns)):
    print(str(X_train.columns[i])+":"+str(imp[i]))
    


#visualize wrong classifications
print("\nVisualize wrong classifications...")
y_test_meta=pd.concat([y_test,X_test_meta],axis=1)

wrongClassified=[]
corrClassified=[]
for i in xrange(len(y_test_meta)):
    if(y_test_meta.iloc[i]['class']!=y_pred_test.iloc[i][0]):
        wrongClassified.append([y_test_meta.iloc[i]])
    elif(y_test_meta.iloc[i]['class']==y_pred_test.iloc[i][0]):
        if( (y_test_meta.iloc[i]['class']=='boat')):
            corrClassified.append([y_test_meta.iloc[i]])
        
print("Wrong detections")
for oneWrong in wrongClassified:
    xmin=oneWrong[0][5]
    ymin=oneWrong[0][6]
    xmax=oneWrong[0][7]
    ymax=oneWrong[0][8]
    path=oneWrong[0][1]
    print(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR) # IMREAD_COLOR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2)
    plt.imshow(img, cmap='gray')
    plt.show()
    print("true label:",oneWrong[0][0])
print("==============================================================================")


