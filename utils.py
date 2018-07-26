def treino_teste_validacao(X,y, frac_train, frac_test):
    import pandas as pd
    #transforma em DataFrames para realizar a divisao
    #Como estamos trabalhando ocm uma serie temporal nao podemos usar fracoes randomicas do dado
    X_all = pd.DataFrame(X)
    y_all = pd.DataFrame(y)

    #Armazena quantos vamos querer guardar
    number_samples_in_train = int(frac_train*X_all.shape[0])
    number_samples_in_test = int(frac_test*X_all.shape[0])

    #Divide treino
    X_train = X_all.head(number_samples_in_train)
    y_train = y_all.iloc[X_train.index]
    
    #Divide teste com o intermediario
    X_test = X_all[number_samples_in_train:number_samples_in_train+number_samples_in_test]
    y_test = y_all.iloc[X_test.index]

    #O resto vira validacao
    X_all = X_all.drop(X_train.index) 
    X_all = X_all.drop(X_test.index)
    X_val = X_all
    y_val = y_all.iloc[X_all.index]

    print('==============')
    print("Tamanho total {}".format(X.shape[0]))
    print("Tamanho treino {}".format(X_train.shape[0]))
    print("Tamanho teste {}".format(X_test.shape[0]))
    print("Tamanho validacao {}".format(X_val.shape[0]))
    print('==============')
    #Retorna resultado como matrizes
    return X_train.as_matrix(),y_train.as_matrix(),\
           X_test.as_matrix(),y_test.as_matrix(),\
           X_val.as_matrix(),y_val.as_matrix()