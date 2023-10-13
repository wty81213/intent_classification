import pandas as pd 
import numpy as np 
from sklearn.metrics import *
from sklearn.metrics import classification_report

result = pd.read_excel('C:\\Users\\cloudy822\\Desktop\\intent classification\\predict\\source\\result_0916_F_v1.xlsx')

real_y = result['問題類別_調整']
# real_y = result['問題類別']
predict_y = result['predict_problem']
sum(real_y!=predict_y)

F1score_macro = f1_score(real_y, predict_y, average = 'macro')
F1score_micro = f1_score(real_y, predict_y, average = 'micro')
Recall_macro = recall_score(real_y, predict_y, average = 'macro')
Recall_micro = recall_score(real_y, predict_y, average = 'micro')
Precision_macro = precision_score(real_y, predict_y, average = 'macro')
Precision_micro = precision_score(real_y, predict_y, average = 'micro')
accuracy = sum(real_y == predict_y)/len(real_y)
#roc_auc_ovo = roc_auc_score(real_y, predict_y, multi_class = 'ovo')        
#roc_auc_ovr = roc_auc_score(real_y, predict_y, multi_class = 'ovr')        

{'F1score_macro':F1score_macro,
 'F1score_micro':F1score_micro,
 'Recall_macro':Recall_macro,
 'Recall_micro':Recall_micro,
 'Precision_macro':Precision_macro,
 'Precision_micro':Precision_micro,
 'accuracy':accuracy}



unique_class = real_y.unique().tolist()
accuracy_for_class = {}
for class_name in unique_class:
    # class_name = unique_class[0]
    real_y_y = real_y == class_name
    predict_y_y = predict_y == class_name

    accuracy = sum(real_y_y == predict_y_y)/len(real_y_y)

    accuracy_for_class[class_name] = np.round(accuracy,4)
accuracy_table = pd.DataFrame(accuracy_for_class,index = ['accuracy']).transpose()

result = classification_report(real_y,predict_y,output_dict=True)
result = pd.DataFrame(result).transpose()

result = result.merge(accuracy_table, how = 'left', left_index=True, right_index=True)

result.to_excel('./temp.xlsx',index = True)
