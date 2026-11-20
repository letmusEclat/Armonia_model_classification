import pandas as pd, numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv('recs.csv',encoding='latin-1')
ENT=['Hair Color','Eye Color','Skin Tone','Under Tone']
T='Recommended Clothing Colors'
full=df.copy()
df=df.drop_duplicates(subset=ENT+[T]).reset_index(drop=True)   # 720 perfiles únicos, igual que el notebook
le=LabelEncoder(); y=le.fit_transform(df[T])
Xtr,Xte,ytr,yte=train_test_split(df[ENT],y,test_size=.2,random_state=42,stratify=y)
oh=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(Xtr)
dt=DecisionTreeClassifier(random_state=42).fit(oh.transform(Xtr),ytr)
print("test acc", dt.score(oh.transform(Xte),yte))
t=dt.tree_
avoid=[df.loc[df[T]==c,'Avoid Clothing Colors'].iloc[0] for c in le.classes_]
model={
 "inputs":ENT,
 "categories":{c:[str(v) for v in cats] for c,cats in zip(ENT,oh.categories_)},
 "featureNames":[str(n) for n in oh.get_feature_names_out()],   # 'Under Tone_Cool'
 "left":t.children_left.tolist(),"right":t.children_right.tolist(),
 "feature":t.feature.tolist(),"threshold":[round(float(v),4) for v in t.threshold],
 "leafClass":[int(np.argmax(v)) for v in t.value[:,0,:]],
 "classes":le.classes_.tolist(),"avoid":avoid,
}
json.dump(model,open('model.json','w'),ensure_ascii=False)
print("nodos:",t.node_count,"| clases:",model["classes"])
# referencia completa para verificar JS: predicción sklearn de las 15120 filas originales
ref=full[ENT].copy(); ref['pred']=dt.predict(oh.transform(full[ENT]))
ref.to_json('ref_rows.json',orient='records')
