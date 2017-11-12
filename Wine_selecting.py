import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle



# Loading the data sets
#--------------------------------------


# loading the red wine data set,printing the top 5 entries and 
# creating a new column in df_red data frame and assign it to 1 for red wine

df_red=pd.read_csv('winequality-red.csv', sep=';')
#print('The top entries for red wine:',df_red.head())
df_red['type']=1 


# repeating the same for white wine (assign type=0 for white wine)
df_white=pd.read_csv('winequality-white.csv', sep=';')
#print('The top entries white wine:',df_white.head())
df_white['type']=0
df_wine=df_red.append(df_white)



def plotting_pair_pot(k,df,column_name):
    
    correlation=df.corr()
    cols=correlation.nlargest(n=k,columns=column_name)[column_name].index # getting highly correlated features
    sns.pairplot(df[cols])

#Plotting a pair plot to understand highest coreelated features with respect to feature 'type'
k=5
#plotting_pair_pot(k,df_wine,'type')   

def max_category_vals(feature_name,df1,df2):
    feature_vals_df1=df1[feature_name].unique()
    feature_vals_df2=df2[feature_name].unique()
    max_category_vals=[]
    if (feature_vals_df1.shape[0] > feature_vals_df2.shape[0]):
        max_category_vals=df1[feature_name].unique()
    else:
        max_category_vals=df2[feature_name].unique()
    #max_category_vals=max(feature_vals_df1.shape[0],feature_vals_df2.shape[0])
    print('The maximum of {} values '.format(feature_name))
    return max_category_vals

def count_category(feature_name,df,max_qulity_vals):
    #feature_vals=df[feature_name].unique()
    
    feature_vals=np.sort(max_qulaity_vals) # sorting values from lower to higher
    count_type=[] # creating a list to count each quality value
    for i in range(len(feature_vals)):
    #print(quality_vals[i])
        count_type.append(df_white[df_white['quality']==feature_vals[i]].shape[0])

    # getting all the quality values from df_wine    
    data={'quality':feature_vals, 'total_count':count_type} # 'stock_dfs/{}.csv'.format(ticker)

    df_type_feature=pd.DataFrame(data)
    return df_type_feature


max_qulaity_vals=max_category_vals('quality',df_white,df_red)
# creating a new data structure for categorical feature 'quality' and 
# see the distribution of wine 'type'
df_white_quality=count_category('quality',df_white,max_qulaity_vals)
df_red_quality=count_category('quality',df_red,max_qulaity_vals)

# plotting a bar chart with the distrivution of the quantity of  each wine type 
# against the 'quality'
'''
p1=plt.bar(df_red_quality['quality'],df_red_quality['total_count'],color='#d62728')
p2=plt.bar(df_white_quality['quality'],df_white_quality['total_count'],
           color='blue',bottom=df_red_quality['total_count'])
plt.xlabel('quality')
plt.ylabel('total count')
plt.legend(['red wine','white wine'])
'''

# plotting 'quality Vs fixed acidity' for each wine type
#-------------------------------------------------------------
'''
plt.scatter( df_red['quality'],df_red['fixed acidity'], color='red')
plt.scatter(df_white['quality'],df_white['fixed acidity'],color='blue', marker='.')
plt.xlabel('Quality')
plt.ylabel('fixed acidity')
plt.legend(['red','white'])
'''

# plotting corelation
#---------------------------

correlation=df_wine.corr()

'''
plt.subplots(figsize=(12,10))
sns.heatmap(correlation,vmax=0.8,square=True)
plt.yticks(rotation=0) # rotate tick labels to clear vicibility
plt.xticks(rotation=90) 
'''




# Printing The most corelated series for 'type'
#print('The correlation series for type:',correlation['type'])
# Printing The most corelated series for 'type'
#print('The correlation series for type:',correlation['type'])

corr_type=correlation['type']
# absolute value of corr type greater than 0.1
corr_type=corr_type[np.abs(corr_type)>.1]
corr_type=corr_type.drop('type') # dropping the 'type' as it's corelated with it self with 1

# assigning values for X and y after shuffeling
df_wine = df_wine.sample(frac=1).reset_index(drop=True)

# getting the highest corelated two features
# -----------------------------------------------------

two_max_corr=(abs(corr_type)).nlargest(n=2)
print("The two most highly correlated features are:%s and %s " % (two_max_corr.index[0],two_max_corr.index[1]))
X=df_wine[two_max_corr.index]
y=df_wine['type'].values

'''
X=df_wine[corr_type.index].values
y=df_wine['type'].values
'''

#Standerdising the X data
# --------------------------------


stdsc=StandardScaler()
X_train_std=stdsc.fit_transform(X)

X=X_train_std

#X=np.reshape(X,(X.shape[1],X.shape[0]))
X=X.T 
y=np.reshape(y,(1,y.shape[0]))# converting rank 1 array to 2D array with nx X m

#X_norm=(X-X.min())/(X.max()-X.min())






# getting the shapes of the data set we have
#----------------------------------------------


shape_X = X.shape
shape_y = y.shape
m = X.shape[1]  # training set size


print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_y))
print ('I have m = %d training examples!' % (m))

def train_test_split(X,y,train_prec):
    #X, y = shuffle(X,y, random_state=0)
    row_nu=int(round(X.shape[1]*train_prec))
    X_train=X[:,:row_nu]
    X_test=X[:,row_nu:]
    y_train=y[:,:row_nu]
    y_test=y[:,row_nu:]
    
    return X_train, X_test,y_train,y_test

# defining the neural network
#---------------------------------------------------


def layer_sizes(X, Y):
    
    
    n_x = X.shape[0] # size of input layer
    n_h = 4 # I choosed this to be approximately double of input nodes
    n_y = Y.shape[0] # size of output layer
    
    return (n_x, n_h, n_y)



def initialize_parameters(n_x, n_h, n_y):
    
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    
   
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
   
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def sigmoid(z):

    s = 1/(1+np.exp(-z))
    
    return s



def forward_propagation(X, parameters):
    
  
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
   
    
    Z1 = np.dot((W1),X)+b1
    A1 = np.tanh(Z1)
  
    Z2 = np.dot((W2),A1)+b2
    A2 = sigmoid(Z2)
    
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y, parameters):
   
    
    m = Y.shape[1] # number of example


    logprobs = (np.multiply(np.log(A2),Y) +np.multiply((np.log(1-A2)),(1-Y)))
    cost = -np.sum(logprobs)/m
    
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost


def backward_propagation(parameters, cache, X, Y):
 
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    ### START CODE HERE ### (≈ 2 lines of code)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    ### END CODE HERE ###
        
    # Retrieve also A1 and A2 from dictionary "cache".
    ### START CODE HERE ### (≈ 2 lines of code)
    A1 = cache["A1"]
    A2 = cache["A2"]
    ### END CODE HERE ###
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    ### START CODE HERE ### (≈ 6 lines of code, corresponding to 6 equations on slide above)
    dZ2 = A2-Y
    dW2 = np.dot(dZ2,(A1.T))/m
    db2 = np.sum(dZ2, axis=1,keepdims=True)/m
    dZ1 = np.multiply(np.dot((W2.T),dZ2),(1-np.power(A1,2)))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m
    ### END CODE HERE ###
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads




def update_parameters(parameters, grads, learning_rate = 1.2):
  

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
  
    

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2

    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def nn_model(X, Y, n_h, num_iterations = 1000, print_cost=False):
   
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    
    # Loop (gradient descent)

    for i in range(0, num_iterations):
         

        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate = 1.2)
        
        ### END CODE HERE ###
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters


def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 >0.5) # this will give a boolean array.For this case this is fine since outputs are 1 and 0.

    return predictions

# end of neural network
#///////////////////////////////////////////////////////////////

X_train, X_test,y_train,y_test=train_test_split(X,y,0.9)
(n_x, n_h, n_y) = layer_sizes(X, y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))


parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

A2, cache= forward_propagation(X, parameters)
# Note: we use the mean here just to make sure that your output matches ours. 
print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))

print("cost = " + str(compute_cost(A2, y, parameters)))

grads = backward_propagation(parameters, cache, X, y)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))

parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

parameters = nn_model(X, y, 4, num_iterations=1000, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))



# Print training  accuracy
#---------------------------

predictions = predict(parameters, X_train)
print("predictions mean = " + str(np.mean(predictions)))
print ('Training Accuracy: %d' % float((np.dot(y_train,predictions.T) + np.dot(1-y_train,1-predictions.T))/float(y_train.size)*100) + '%')



# Print testing accuracy
#---------------------------

predictions = predict(parameters, X_test)
print("predictions mean = " + str(np.mean(predictions)))
print ('Testing Accuracy: %d' % float((np.dot(y_test,predictions.T) + np.dot(1-y_test,1-predictions.T))/float(y_test.size)*100) + '%')

# Plotting the decision boundry
# ----------------------------------
def plotting_decision_boundry(X,y):
    X=X.T
    y=y.T
    markers=('x','x')
    colors=('red','blue')
    cmap=ListedColormap(colors[:len(np.unique(y))])
    #X=transformed.values
    resolution=0.02
    x1_min,x1_max =X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min, x2_max,resolution))

    Z = predict(parameters,np.c_[xx1.ravel(), xx2.ravel()].T)
    Z=Z.reshape(xx1.shape)

    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[:,0,np.newaxis][y==cl,np.newaxis],X[:,1,np.newaxis][y==cl,np.newaxis],alpha= 0.8, c=cmap(idx),marker=markers[idx],label=cl)
    
plotting_decision_boundry(X,y)
plt.xlabel('total sulfur dioxide')
plt.ylabel('volatile acidity')
#plt.legend(['red wine','white wine'],locals)
plt.legend(['red wine','white wine'],bbox_to_anchor=(1.05, 1), loc=4, borderaxespad=0.)
plt.show()

# increasing the number of features 
# Using all the feartures and PCA 
#------------------------------------------------

from sklearn.decomposition import PCA as sklearnPCA
X=df_wine[corr_type.index].values
y=df_wine['type'].values
pca=sklearnPCA(n_components=2)
X_norm=(X-X.min())/(X.max()-X.min())
transformed=pd.DataFrame(pca.fit_transform(X_norm))
X=transformed.values