#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def randomforest(num):
    data = pd.read_csv('data.csv')
    print(data)
    
    x = data.iloc[:, 3:4].values 
    #print("x is", x)
    y = data.iloc[:, 4].values  
    #print("y is",y)
    # Fitting Random Forest Regression to the dataset
    # import the regressor
    from sklearn.ensemble import RandomForestRegressor
      
      # create regressor object
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
      
    # fit the regressor with x and y data
    regressor.fit(x, y)  
    
    pred1 = regressor.predict(np.array([num]).reshape(1, 1))  # test the output by changing values
    #print(Y_pred)
    print("Random Forest output is",pred1)
    
    
    # Visualising the Random Forest Regression results
      
    # arange for creating a range of values
    # from min value of x to max 
    # value of x with a difference of 0.01 
    # between two consecutive values
    X_grid = np.arange(min(x), max(x), 0.01) 
      
    # reshape for reshaping the data into a len(X_grid)*1 array, 
    # i.e. to make a column out of the X_grid value                  
    X_grid = X_grid.reshape((len(X_grid), 1))
      
    # Scatter plot for original data
    plt.scatter(x, y, color = 'blue')  
      
    # plot predicted data
    plt.plot(X_grid, regressor.predict(X_grid), 
              color = 'green') 
    plt.title('Random Forest Regression')
    plt.xlabel('Max Selling Price')
    plt.ylabel('Seling Price')
    plt.savefig('randomforest.jpg')
    print(plt.show())
    return pred1


def decisiontree(num):
    #import numpy package for arrays and stuff
    import numpy as np 
      
    # import matplotlib.pyplot for plotting our result
    import matplotlib.pyplot as plt
      
    # import pandas for importing csv files 
    import pandas as pd 
    
    
    data = pd.read_csv('data.csv') 
    
    x = data.iloc[:, 3:4].values 
    #print("x is", x)
    y = data.iloc[:, 4].values  
    #print("y is",y)
    
    
    #import the regressor
    from sklearn.tree import DecisionTreeRegressor 
      
    # create a regressor object
    regressor = DecisionTreeRegressor(random_state = 0) 
      
    # fit the regressor with X and Y data
    regressor.fit(x, y)
    
    
    # predicting a new value
      
    # test the output by changing values, like 3750
    pred2 = regressor.predict([[num]])
      
    # print the predicted price
    #print("Predicted price: % d\n"% y_pred) 
    print("Decision tree output is",pred2)
    
    
    # arange for creating a range of values 
    # from min value of X to max value of X 
    # with a difference of 0.01 between two
    # consecutive values
    X_grid = np.arange(min(x), max(x), 0.01)
      
    # reshape for reshaping the data into 
    # a len(X_grid)*1 array, i.e. to make
    # a column out of the X_grid values
    X_grid = X_grid.reshape((len(X_grid), 1)) 
      
    # scatter plot for original data
    plt.scatter(x, y, color = 'red')
      
    # plot predicted data
    plt.plot(X_grid, regressor.predict(X_grid), color = 'blue') 
      
    # specify title
    plt.title('Sell Price Prediction (Decision Tree Regression)') 
      
    # specify X axis label
    plt.xlabel('maxtrade cost')
      
    # specify Y axis label
    plt.ylabel('selling price')
      
    #show the plot
    plt.savefig('decisiontree.jpg')
    plt.show()
    
    
    
    # import export_graphviz
    from sklearn.tree import export_graphviz 
      
    # export the decision tree to a tree.dot file
    # for visualizing the plot easily anywhere
    export_graphviz(regressor, out_file ='tree.dot',
                    feature_names =['Production Cost']) 
    return pred2



def xgboost(num):
    print("num is ",num)
    # Necessary imports
    import numpy as np
    import pandas as pd
    import xgboost as xg
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error as MSE
      
    # Load the data
    data = pd.read_csv("data.csv")
    #X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
    
    x = data.iloc[:, 3:4].values 
    #print("x is", x)
    y = data.iloc[:, 4].values  
    #print("y is",y)
    
      
    # Splitting
    train_X, test_X, train_y, test_y = train_test_split(x, y,
                          test_size = 0.3, random_state = 123)
    
    #print(test_X)
      
    # Instantiation
    xgb_r = xg.XGBRegressor(objective ='reg:linear',
                      n_estimators = 10, seed = 123)
      
    # Fitting the model
    xgb_r.fit(train_X, train_y)
      
    # Predict the model
    num=int(num)
    b = np.array([num])
    pred3 = xgb_r.predict(b)
    
    print("xg boost output is",pred3)
      
    # # RMSE Computation
    # rmse = np.sqrt(MSE(test_y, pred))
    # print("RMSE : % f" %(rmse))
    
    
    # arange for creating a range of values 
    # from min value of X to max value of X 
    # with a difference of 0.01 between two
    # consecutive values
    X_grid = np.arange(min(x), max(x), 1000)
      
    # reshape for reshaping the data into 
    # a len(X_grid)*1 array, i.e. to make
    # a column out of the X_grid values
    X_grid = X_grid.reshape((len(X_grid), 1)) 
      
    # scatter plot for original data
    plt.scatter(x, y, color = 'yellow')
      
    # plot predicted data
    plt.plot(X_grid, xgb_r.predict(X_grid), color = 'red') 
      
    # specify title
    plt.title('Sell Price Prediction (XGboost Regression)') 
      
    # specify X axis label
    plt.xlabel('maxtrade cost')
      
    # specify Y axis label
    plt.ylabel('selling price')
      
    # show the plot
    plt.savefig('xgboost.jpg')
    plt.show()
    return pred3






 
def main(num):
    pred1=randomforest(num)
    pred2=decisiontree(num)
    pred3=xgboost(num)
    
    # creating the dataset
    import pandas as pd
    from pandas import Series, DataFrame
    import matplotlib.pyplot as plt
    
    data = [pred1,pred2,pred3]
    labels = ['random forest', 'decision tree', 'xgboost']
    plt.xticks(range(len(data)), labels)
    plt.xlabel('ML Algorithm')
    plt.ylabel('selling price')
    plt.title('Arecanaut price prediction')
    plt.bar(range(len(data)), data) 
    plt.savefig('final.jpg')
    plt.show()
    return pred1,pred2,pred3
    


#xgboost(28599)