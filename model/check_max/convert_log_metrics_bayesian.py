import numpy as np
import pandas as pd
df = pd.read_csv('run_log.txt')


con_loss =[]
con_trrain_acc =[]
con_train_miou =[]
con_val_acc=[]
con_val_miou=[]
con_kl_div=[]

file = np.array(df)
for row in file:
    # print(row)
    if row[0].startswith('loss:'):
        con_loss.append(row[0].split()[-1])
    if row[0].startswith('train_accuracy:'):
        con_trrain_acc.append(row[0].split()[-1])    
    if row[0].startswith('train_miou:'):
        con_train_miou.append(row[0].split()[-1])        
    if row[0].startswith('val_accuracy:'):
        con_val_acc.append(row[0].split()[-1])
    if row[0].startswith('val_miou:'):
        con_val_miou.append(row[0].split()[-1])
    if row[0].startswith('kl divergence:'):
        con_kl_div.append(row[0].split()[-1])                  

np.savetxt('train_acc_container.txt', con_trrain_acc,fmt='%s')
np.savetxt('train_loss_container.txt', con_loss,fmt='%s')
np.savetxt('train_miou_container.txt', con_train_miou,fmt='%s')
np.savetxt('val_acc_container.txt', con_val_acc,fmt='%s')
np.savetxt('val_miou_container.txt', con_val_miou,fmt='%s')
np.savetxt('con_kl_div.txt', con_kl_div,fmt='%s')
