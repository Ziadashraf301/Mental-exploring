from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

def get_report(
    model,
    x_train,
    y_train,
    x_test,
    y_test):
  
  y_pred_train = model.predict(x_train)
  y_pred_test= model.predict(x_test)

  acc_train = accuracy_score(y_train,y_pred_train)
  acc_test = accuracy_score(y_test,y_pred_test)

  f1_train = f1_score(y_train,y_pred_train,average='macro')
  f1_test = f1_score(y_test,y_pred_test,average='macro')

  prob_train = model.predict_proba(x_train)
  prob_test = model.predict_proba(x_test)

  loss_train = log_loss(y_train, prob_train)
  loss_test = log_loss(y_test, prob_test)

  print('Loss Of Train : %0.2f'%(loss_train))
  print('Loss Of Test : %0.2f'%(loss_test))
  print('Acuuracy Train : %0.2f'%(acc_train*100))
  print('Acuuracy Test : %0.2f'%(acc_test*100))
  print('F1 Train : %0.2f'%(f1_train*100))
  print('F1 Test : %0.2f'%(f1_test*100))
  print(classification_report(y_test, y_pred_test, target_names= ['Sad', 'Happy']))
