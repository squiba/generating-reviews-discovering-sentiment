import pandas as pd 
import numpy as np 

from encoder import Model 

df = pd.read_csv('questions.csv')
question1 = np.array(df['question1'])
question2 = np.array(df['question2'])
labels = np.array(df['is_duplicate'])

del df

model = Model()
for i in range(21,40):
	ques1_features = model.transform(question1[10000*i:10000*(i+1)])
	ques2_features = model.transform(question2[10000*i:10000*(i+1)])
	label = labels[10000*i:10000*(i+1)].reshape([10000,1])

	data = np.concatenate((ques1_features,ques2_features,label),axis=1)
	np.save("quora_data/quora_features{}".format(i),data)

	del ques1_features,ques2_features,label,data