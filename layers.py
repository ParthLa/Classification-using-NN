'''This file contains the implementations of the layers required by your neural network

For each layer you need to implement the forward and backward pass. You can add helper functions if you need, or have extra variables in the init function

Each layer is of the form - 
class Layer():
    def __init__(args):
        *Initializes stuff*

    def forward(self,X):
        # X is of shape n x (size), where (size) depends on layer
        
        # Do some computations
        # Store activations_current
        return X

    def backward(self, lr, activation_prev, delta):
        """
        # lr - learning rate
        # delta - del_error / del_activations_current
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        """
        # Compute gradients wrt trainable parameters
        # Update parameters
        # Compute gradient wrt input to this layer
        # Return del_error/del_activation_prev
'''
import numpy as np

class FullyConnectedLayer:
    def __init__(self, in_nodes, out_nodes, activation):
        # Method to initialize a Fully Connected Layer
        # Parameters
        # in_nodes - number of input nodes of this layer
        # out_nodes - number of output nodes of this layer
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.activation = activation   # string having values 'relu' or 'softmax', activation function to use
        # Stores the outgoing summation of weights * feautres 
        self.data = None

        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))    
        self.biases = np.random.normal(0,0.1, (1, out_nodes))
        ###############################################
        # NOTE: You must NOT change the above code but you can add extra variables if necessary 

    def forwardpass(self, X):
        '''
                
        Arguments:
            X  -- activation matrix       :[n X self.in_nodes]
        Return:
            activation matrix      :[n X self.out_nodes]
        '''
        # TODO
        # print(np.shape(X))
        # print(np.shape(self.weights))
        Y=np.matmul(X,self.weights) # [Y]=[n X self.out_nodes]
        Y=Y+self.biases
        self.data=Y

        if self.activation == 'relu':
            Y=relu_of_X(Y)
            # raise NotImplementedError
        elif self.activation == 'softmax':
            Y=softmax_of_X(Y)
            # raise NotImplementedError

        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        # self.data=Y
        return Y
        # pass
        # END TODO      
    def backwardpass(self, lr, activation_prev, delta):
        '''
        # lr - learning rate
        # delta - del_error / del_activations_current  : 
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        '''

        # TODO 
        # print(np.shape(activation_prev)) # [activation_prev]=[n X self.in_nodes]
        Z=np.matmul(activation_prev,self.weights) # [Z]=[n X self.out_nodes]
        n=Z.shape[0]
        Z=Z+self.biases # Z = current_data
        if self.activation == 'relu':
            Y=gradient_relu_of_X(Z,delta) # [Y]=[n X self.out_nodes]
        elif self.activation == 'softmax':
            Y=gradient_softmax_of_X(Z,delta) # [Y]=[n X self.out_nodes]

        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        delta_prev = np.matmul(Y,self.weights.transpose()) # [delta_prev]=[n X out]

        self.weights=self.weights - lr*np.matmul(activation_prev.transpose(),Y)/n # [W]=[in X out]
        # self.biases=self.biases - lr*np.matmul(np.ones((1,n)),Y)/n # [b]=[1 X out]
        self.biases=self.biases - lr*np.mean(Y,axis=0).reshape([1,-1]) # /n done in mean itself
        
        return delta_prev
        
        # END TODO
class ConvolutionLayer:
    def __init__(self, in_channels, filter_size, numfilters, stride, activation):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for convolution layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer
        # numfilters  - number of feature maps (denoting output depth)
        # stride      - stride to used during convolution forward pass
        # activation  - can be relu or None
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride
        self.activation = activation
        self.out_depth = numfilters
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

        # Stores the outgoing summation of weights * feautres 
        self.data = None
        
        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))   
        self.biases = np.random.normal(0,0.1,self.out_depth)
        

    def forwardpass(self, X):
        # INPUT activation matrix       :[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix      :[n X self.out_depth X self.out_row X self.out_col]

        # TODO
        n=X.shape[0]
        self.in_depth, self.in_row, self.in_col = X.shape[1:]
        s=self.stride
        r=self.filter_row
        c=self.filter_col

        A=np.zeros([n, self.out_depth, self.out_row, self.out_col])
        # i=0
        # j=0
        # for i in range(n):
        #     for j in range(self.out_depth):
        #         for k in range(self.in_depth):
        #             in_matrix=X[i][k]
        #             Filter=self.weights[j][k]
        #             B=np.zeros((self.out_row, self.out_col))
        #             for f in range(self.out_row):
        #                 for g in range(self.out_col):
        #                     i_index=f*self.stride
        #                     j_index=g*self.stride
        #                     T1=in_matrix[i_index:i_index+self.filter_row,j_index:j_index+self.filter_col]
        #                     B[f][g]=np.sum(T1*Filter) # B[f][g] = scalar
        #             A[i][j]+=B # [B] = self.out_row X self.out_col

        for f in range(self.out_row):
            for g in range(self.out_col):
                X_in=X[:,:,f*s:f*s+r,g*s:g*s+c]
                flattened_part = X_in.reshape([n, self.in_depth, -1])
                # V = self.weights.reshape(list(self.weights.shape[:2]) + [-1]) # doubt
                V = self.weights.reshape([self.out_depth,self.in_depth,-1])
                A[:, :, f, g] = np.sum(np.einsum('nid,oid->nod', flattened_part,V), axis=-1) # doubt


        # for i in range(n):
        #     for f in range(self.out_row):
        #         for g in range(self.out_col):
        #             A[i,:,f,g]+=self.biases
        A = A + self.biases[np.newaxis,:,np.newaxis,np.newaxis]
        self.data=A

        if self.activation == 'relu':
            for i in range(n):
                for j in range(self.out_depth):
                    A[i,j,:,:]=relu_of_X(A[i,j,:,:])
            self.data=A
            return A

        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        
        ###############################################
        # END TODO
    def backwardpass(self, lr, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # Update self.weights and self.biases for this layer by backpropagation
        # TODO

        ###############################################
        n=activation_prev.shape[0]
        s=self.stride
        r=self.filter_row
        c=self.filter_col

        delta_prev=np.zeros([n,self.in_depth,self.in_row,self.in_col])
        # print(np.shape(activation_prev)) 
        # print(np.shape(delta_prev)) # same as above
        # print(np.shape(self.data))
        # print(np.shape(delta)) # same as above
        inp_delta=None
        if self.activation == 'relu':
            # inp_delta = actual_gradient_relu_of_X(self.data, delta)
            inp_delta = gradient_relu_of_X(self.data, delta) # [inp_delta] = n x out_depth x out_row x out_col
            # raise NotImplementedError
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        for f in range(self.out_row):
            for g in range(self.out_col):
                conv_back = np.einsum('no,oirc->nirc', inp_delta[:,:,f,g], self.weights) # doubt
                delta_prev[:,:,f*s:f*s+r,g*s:g*s+c] += conv_back # doubt

        for f in range(self.out_row):
            for g in range(self.out_col):
                X_in=activation_prev[:,:,f*s:f*s+r,g*s:g*s+c]
                # print(np.shape(X_in))
                conv=inp_delta[:,:,f,g].reshape([n,self.out_depth])
                # print(np.shape(conv))
                self.weights = self.weights - lr*np.mean(np.einsum('no,nirc->noirc', conv, X_in), axis=0) # doubt
                # del_Ew = conv*X_in
                # self.weights = self.weights - lr*np.mean(del_Ew,axis=0)

        T = inp_delta.reshape([n,self.out_depth,-1])
        self.biases = self.biases - lr*np.mean(np.sum(T, axis=-1), axis=0) # [bias] = out_depth x 1

        return delta_prev
        ###############################################

        # END TODO
    
class AvgPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
       	n=X.shape[0]
        self.in_depth, self.in_row, self.in_col = X.shape[1:]
        s=self.stride
        r=self.filter_row
        c=self.filter_col
        A=np.zeros([n, self.out_depth, self.out_row, self.out_col])

        filter_matrix=np.ones([r,c])/(r*c)
        for f in range(self.out_row):
            for g in range(self.out_col):
                T1=X[:,:,f*s:f*s+r,g*s:g*s+c]
                V=T1*filter_matrix[np.newaxis,np.newaxis,:,:]
                A[:,:,f,g]=np.sum(np.sum(V,axis=-1),axis=-1)

        self.data=A
        return A

        # i=0
        # j=0
        # for i in range(n):
        #     for j in range(self.in_depth):
        #         # for k in range(self.in_depth):
        #         in_matrix=X[i][j] # plate in the jth layer of ith element
        #         # Filter=self.weights[i][j] 
        #         B=np.zeros((self.out_row, self.out_col))
        #         for f in range(self.out_row):
        #             for g in range(self.out_col):
        #                 T1=in_matrix[:,:,f*s:f*s+r,g*s:g*s+c]
        #                 T1=X[:,:,f*s:f*s+r,g*s:g*s+c]
        #                 V=T1*filter_matrix[np.newaxis,np.newaxis,:,:]
        #                 B[f][g]=np.sum(in_matrix)/(self.filter_row*self.filter_col) # B[f][g] = scalar
        #         A[i][j]=B # [B] = self.out_row X self.out_col
        
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        delta_prev=np.zeros(np.shape(activation_prev))
        
        s=self.stride
        r=self.filter_row
        c=self.filter_col
        avg=np.ones([r,c])/(r*c) # [avg] = filter_row x filter_col

        for f in range(self.out_row):
            for g in range(self.out_col): 
               # [delta_prev] = n x in_depth x in_row x in_col
               # delta_prev[:,:,f*s:f*s+r,g*s:g*s+c] = delta[:,:][f][g]*avg[np.newaxis,np.newaxis,:,:]
               # why doesn't the above work ?
               delta_prev[:,:,f*s:f*s+r,g*s:g*s+c] = delta[:,:,f:f+1,g:g+1]*avg[np.newaxis,np.newaxis,:,:]

        return delta_prev # [delta_prev] = n x in_depth x in_row x in_col
        # pass
        # END TODO
        ###############################################



class MaxPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input | n x in_depth x in_row x in_col
        # Output
        # activations : Activations after one forward pass through this layer | | n x in_depth x out_row x out_col
        
        # TODO
        n=X.shape[0]
        s=self.stride
        r=self.filter_row
        c=self.filter_col
        self.in_depth, self.in_row, self.in_col = X.shape[1:]
        # Z=np.zeros(np.shape(X))
        A=np.zeros([n, self.out_depth, self.out_row, self.out_col])

        # filter_matrix=np.ones((r,c))
        for f in range(self.out_row):
            for g in range(self.out_col):
                T1=X[:,:,f*s:f*s+r,g*s:g*s+c]
                # T2=np.zeros()
                # V=T1*filter_matrix[np.newaxis,np.newaxis,:,:]
                val=np.max(np.max(T1,axis=-1),axis=-1)
                A[:,:,f,g]=val
                # np.where(T1=val,1,0)

        self.data=A
        return A

        # n=X.shape[0]
        # self.in_depth, self.in_row, self.in_col = X.shape[1:]
        # A=np.zeros([n, self.in_depth, self.out_row, self.out_col])
        # i=0
        # j=0
        # for i in range(n):
        #     for j in range(self.in_depth):
        #         # for k in range(self.in_depth):
        #         in_matrix=X[i][j] # plate in the jth layer of ith element
        #         # Filter=self.weights[i][j] 
        #         B=np.zeros((self.out_row, self.out_col))
        #         for f in range(self.out_row):
        #             for g in range(self.out_col):
        #                 i_index=f*self.stride
        #                 j_index=g*self.stride
        #                 T1=in_matrix[i_index:i_index+self.filter_row,j_index:j_index+self.filter_col]
        #                 B[f][g]=np.max(in_matrix) # B[f][g] = scalar
        #         A[i][j]=B # [B] = self.out_row X self.out_col
        # return A
        # pass
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        # S=self.forwardpass(activation_prev) # [S] = | n x in_depth x out_row x out_col
        # X=np.zeros(np.shape(activation_prev))

        delta_prev=np.zeros(np.shape(activation_prev))
        
        s=self.stride
        r=self.filter_row
        c=self.filter_col
        # mxm=np.zeros([r,c]) # [mxm] = filter_row x filter_col
        # mxm[r//2][c//2]=1
        # print(mxm)

        for f in range(self.out_row):
            for g in range(self.out_col): 
               actv_prev_part = activation_prev[:,:,f*s:f*s+r,g*s:g*s+c]
               max_part = np.max(np.max(actv_prev_part,axis=-1),axis=-1)
               mxm_filter = actv_prev_part==max_part[:,:,np.newaxis,np.newaxis] # [mxm_filter] = [actv_prev]
               # [delta_prev] = n x in_depth x in_row x in_col
               # delta_prev[:,:,f*s:f*s+r,g*s:g*s+c] = delta[:,:,f:f+1,g:g+1]*mxm[np.newaxis,np.newaxis,:,:]
               delta_prev[:,:,f*s:f*s+r,g*s:g*s+c] = delta[:,:,f:f+1,g:g+1]*mxm_filter

        return delta_prev # [delta_prev] = n x in_depth x in_row x in_col
        # pass
        # END TODO
        ###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        # TODO
        # print(X.shape)
        n,d,r,c=X.shape
        Y=np.reshape(X,[n,d*r*c])
        return Y
        # pass

    def backwardpass(self, lr, activation_prev, delta):
        A=np.shape(activation_prev)
        delta_prev=np.reshape(delta,A)
        return delta_prev
        # pass
        # END TODO

# Function for the activation and its derivative
def relu_of_X(X):

    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu
    # TODO
    Y=np.where(X>0,X,0)
    return Y
    # raise NotImplementedError
    # END TODO 
    
def gradient_relu_of_X(X, delta):
    # Input
    # Note that these shapes are specified for FullyConnectedLayers, the function also needs to work with ConvolutionalLayer
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes # term1 of the backprop derivation
    # Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu amd during backwardpass
    
    # TODO
    Y=np.where(X>0,1,0) # this is term2 of the backprop derivation
    # Y=np.dot(Y,delta)
    Y=Y*delta
    return Y # term1*term2 only => del(E)/del(sum) i.e. doesn't include term3 

    # raise NotImplementedError
    # END TODO

def softmax_of_X(X):
    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax
    
    # TODO
    # softmax = exp(xij)/sum_j(exp(xij))
    out=X.shape[1]
    F=np.exp(X) # Fij = e^xij
    Z=np.ones((out,out)) 
    Z=np.matmul(F,Z) # Zij = sum_j(exp(xij))
    Y=np.divide(F,Z)  # Yij = exp(xij)/sum_j(exp(xij))
    # print(X)
    # print(np.sum(Y,axis=1))
    # print(Y)
    return Y
    # raise NotImplementedError
    # END TODO  
def gradient_softmax_of_X(X, delta):
    # Input
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax amd during backwardpass
    # Hint: You might need to compute Jacobian first

    # TODO
    batchSize=X.shape[0]
    out=X.shape[1]
    S=softmax_of_X(X) # [S]= batchSize x self.out_nodes
    Y=np.zeros((batchSize,out))
    for i in range(batchSize):
    	V=S[i,:] # [V] = 1 x self.out_nodes
    	# print(np.shape(V))
    	# T=np.matmul(V.transpose(),V) # [T] = self.out_nodes x self.out_nodes
    	T=np.outer(V,V)
    	U=np.diag(V) # [U] = self.out_nodes x self.out_nodes
    	T=U-T
    	Z=np.matmul(T,delta[i,:].transpose()) # [Z] = self.out_nodes x 1
    	Z=Z.transpose() # [Z] = 1 x self.out_nodes
    	Y[i,:]=Z

    return Y
    # del(sm(X)_11)/del(X_11)=sm(X)_11(1-sm(X)_11) # wrong
    # raise NotImplementedError
    # END TODO
