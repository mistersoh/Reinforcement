import numpy as np
import numpy.matlib 
import matplotlib.pyplot as plt
import pandas as pd
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from features import *
from generate_game import *
from Q_values import *


size_board = 4


"""
beta = decaying trend of epsilon
gamma = discount factor
episodes = number of episodes run
alg_type type of algorithm 0 = q-learning, 1 = SARSA
rms_use = usage of RMSProp 0 = deactivate, 1 = activate
"""
def main(beta, gamma, episodes, alg_type, rms_use):
    """
    Generate a new game
    The function below generates a new chess board with King, Queen and Enemy King pieces randomly assigned so that they
    do not cause any threats to each other.
    s: a size_board x size_board matrix filled with zeros and three numbers:
    1 = location of the King
    2 = location of the Queen
    3 = location fo the Enemy King
    p_k2: 1x2 vector specifying the location of the Enemy King, the first number represents the row and the second
    number the colunm
    p_k1: same as p_k2 but for the King
    p_q1: same as p_k2 but for the Queen
    """
    s, p_k2, p_k1, p_q1 = generate_game(size_board)

    """
    Possible actions for the Queen are the eight directions (down, up, right, left, up-right, down-left, up-left, 
    down-right) multiplied by the number of squares that the Queen can cover in one movement which equals the size of 
    the board - 1
    """
    possible_queen_a = (s.shape[0] - 1) * 8
    """
    Possible actions for the King are the eight directions (down, up, right, left, up-right, down-left, up-left, 
    down-right)
    """
    possible_king_a = 8

    # Total number of actions for Player 1 = actions of King + actions of Queen
    N_a = possible_king_a + possible_queen_a

    """
    Possible actions of the King
    This functions returns the locations in the chessboard that the King can go
    dfK1: a size_board x size_board matrix filled with 0 and 1.
          1 = locations that the king can move to
    a_k1: a 8x1 vector specifying the allowed actions for the King (marked with 1): 
          down, up, right, left, down-right, down-left, up-right, up-left
    """
    dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
    """
    Possible actions of the Queen
    Same as the above function but for the Queen. Here we have 8*(size_board-1) possible actions as explained above
    """
    dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
    """
    Possible actions of the Enemy King
    Same as the above function but for the Enemy King. Here we have 8 possible actions as explained above
    """
    dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

    """
    Compute the features
    x is a Nx1 vector computing a number of input features based on which the network should adapt its weights  
    with board size of 4x4 this N=50
    """
    x = features(p_q1, p_k1, p_k2, dfK2, s, check)
    [p] = x.shape

    """
    Initialization
    Define the size of the layers and initialization
    
    Define the network, the number of the nodes of the hidden layer should be 200. The weights 
    should be initialised according to a uniform distribution and rescaled by the total number of connections between 
    the considered two layers. The biases should be initialized with zeros.
    """
    n_input_layer = p  # Number of neurons of the input layer. TODO: Change this value
    n_hidden_layer = 200  # Number of neurons of the hidden layer
    n_output_layer = 32  # Number of neurons of the output layer. TODO: Change this value accordingly

    """
    Define the w weights between the input and the hidden layer and the w weights between the hidden layer and the 
    output layer according to the instructions. Define also the biases.
    """


    """
    Weights W1 = Input => hidden, W2 = hidden => output
    """
    W1=np.random.uniform(0,1,(n_hidden_layer,n_input_layer));
    W1=np.divide(W1,np.matlib.repmat(np.sum(W1,1)[:,None],1,n_input_layer));

    W2=np.random.uniform(0,1,(n_output_layer,n_hidden_layer));
    W2=np.divide(W2,np.matlib.repmat(np.sum(W2,1)[:,None],1,n_hidden_layer));


    """
    Biases bias_W1 = Input => hidden, bias_W2 = hidden => output
    """
    bias_W1=np.zeros((n_hidden_layer,))
    bias_W2=np.zeros((n_output_layer,))


    # Network Parameters
    epsilon_0 = 0.2         #epsilon for the e-greedy policy
    beta = beta             #epsilon discount factor (0.00005)
    gamma = gamma           #SARSA Learning discount factor (0.85)
    eta = 0.0035            #learning rate
    N_episodes = episodes   #Number of games, each game ends when we have a checkmate or a draw (100000)
    alpha = 1/10000         #Smoothing rate for exponential moving average
    grad_squared = 0.0      #initial gradient for the RMSProp

    ###  Training Loop  ###

    # Directions: down, up, right, left, down-right, down-left, up-right, up-left
    # Each row specifies a direction, 
    # e.g. for down we need to add +1 to the current row and +0 to current column
    map = np.array([[1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                    [1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1]])
    
    # THE FOLLOWING VARIABLES COULD CONTAIN THE REWARDS PER EPISODE AND THE
    # NUMBER OF MOVES PER EPISODE, FILL THEM IN THE CODE ABOVE FOR THE
    # LEARNING. OTHER WAYS TO DO THIS ARE POSSIBLE, THIS IS A SUGGESTION ONLY.    

    R_save = np.zeros([N_episodes, 1])
    N_moves_save = np.zeros([N_episodes, 1])
    N_moves_ema = np.zeros([N_episodes+1, 1])
    S_ema = np.zeros([N_episodes+1, 1])
    # END OF SUGGESTIONS
    

    for n in range(N_episodes):
        epsilon_f = epsilon_0 / (1 + beta * n) #epsilon is discounting per iteration to have less probability to explore
        checkmate = 0  # 0 = not a checkmate, 1 = checkmate
        draw = 0  # 0 = not a draw, 1 = draw
        i = 1  # counter for movements

        # Generate a new game
        s, p_k2, p_k1, p_q1 = generate_game(size_board)

        # Possible actions of the King
        dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
        # Possible actions of the Queen
        dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
        # Possible actions of the enemy king
        dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)
        
        
        #####################################
        
        hiddenOld = np.zeros((n_hidden_layer,1))
        inputOld = np.zeros((n_input_layer,1))
        rectOut = np.zeros((n_output_layer,1))
        QvalueOld = 0
        rOld = 0
        
        #####################################

        while checkmate == 0 and draw == 0:
            R = 0  # Reward

            # Player 1

            # Actions & allowed_actions
            a = np.concatenate([np.array(a_q1), np.array(a_k1)])
            allowed_a = np.where(a > 0)[0]

            # Computing Features
            x = features(p_q1, p_k1, p_k2, dfK2, s, check)

            nX = np.zeros((n_input_layer,1))
            for i in range(n_input_layer):
                nX[i,0] = x[i]
            # Fill the Q using computed Q values as output of neural network.
            Q, out1 = Q_values(x, W1, W2, bias_W1, bias_W2)
            
            nOut1 = np.zeros((n_hidden_layer,1))
            for i in range(n_hidden_layer):
                nOut1[i,0] = out1[i]
            
            # Epsilon-greedy parameter
            eGreedy = int(np.random.rand() < epsilon_f)
            
   
            if(eGreedy):
                action = np.random.choice(allowed_a)
            else:
                argMax = np.argmax(Q[allowed_a])
                action = allowed_a[argMax]
             

            a_agent = action  # TO USE EPSILON GREEDY POLICY

            # Player 1 makes the action
            if a_agent < possible_queen_a:
                direction = int(np.ceil((a_agent + 1) / (size_board - 1))) - 1
                steps = a_agent - direction * (size_board - 1) + 1

                s[p_q1[0], p_q1[1]] = 0
                mov = map[direction, :] * steps
                s[p_q1[0] + mov[0], p_q1[1] + mov[1]] = 2
                p_q1[0] = p_q1[0] + mov[0]
                p_q1[1] = p_q1[1] + mov[1]

            else:
                direction = a_agent - possible_queen_a
                steps = 1

                s[p_k1[0], p_k1[1]] = 0
                mov = map[direction, :] * steps
                s[p_k1[0] + mov[0], p_k1[1] + mov[1]] = 1
                p_k1[0] = p_k1[0] + mov[0]
                p_k1[1] = p_k1[1] + mov[1]

            # Compute the allowed actions for the new position

            # Possible actions of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

            # Player 2

            # Check for draw or checkmate
            if np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 1:
                # King 2 has no freedom and it is checked
                # Checkmate and collect reward
                checkmate = 1
                R = 1  # Reward for checkmate

                """
                Update the parameters of network by applying backpropagation and Q-learning. Use the 
                rectified linear function as activation function. Exploit the Q value for the action made.
                This is the last iteration of the episode, the agent gave checkmate.
                """


                if alg_type == 0:

                    """ Q-Learning """
                
                    # Rectified output
                    output = np.zeros((n_output_layer,1))
                    output[a_agent,0] = 1
                    
                    if rms_use == 0:
                        # Weighted updated for each iteration
                        dw2 = eta * (rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut.dot(hiddenOld.T)
                        W2 += dw2
                        
                        dbw2 = eta * (rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut.flatten('F')
                        bias_W2 += dbw2
                        
                        # Heaviside function
                        r_Hidden = np.heaviside(hiddenOld, 0)
                        
                        dw1 = eta * (((rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut).T.dot(dw2).T * r_Hidden).dot(inputOld.T)
                        W1 += dw1
                        
                        dbw1 = (eta * ((rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut).T.dot(dw2).T * r_Hidden).flatten('F')
                        bias_W1 += dbw1
                        
                        # Update variables for q-learning
                        QvalueOld = np.max(Q[allowed_a])
                        rectOut = output
                        hiddenOld = nOut1
                        inputOld = nX
                        rOld = R
                        
                        
                    else:
                    
                        # Weighted updated for each iteration
                        dw2 = eta * (rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut.dot(hiddenOld.T)
                        gradient = np.gradient(dw2, axis=0)
                        gradient = np.array(gradient, dtype=float)
                        grad_squared = 0.9 * grad_squared + 0.1
                        dw2 = dw2 - (eta/np.sqrt(grad_squared)) * gradient
                        W2 += dw2
                        
                        dbw2 = eta * (rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut.flatten('F')
                        gradient = np.gradient(dbw2, axis=0)
                        gradient = np.array(gradient, dtype=float)
                        grad_squared = 0.9 * grad_squared + 0.1
                        dbw2 = dbw2 - (eta/np.sqrt(grad_squared)) * gradient
                        bias_W2 += dbw2
                        
                        # Heaviside function
                        r_Hidden = np.heaviside(hiddenOld, 0)
                        
                        dw1 = eta * (((rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut).T.dot(dw2).T * r_Hidden).dot(inputOld.T)
                        gradient = np.gradient(dw1, axis=0)
                        gradient = np.array(gradient, dtype=float)
                        grad_squared = 0.9 * grad_squared + 0.1
                        dw1 = dw1 - (eta/np.sqrt(grad_squared)) * gradient
                        W1 += dw1
                        
                        dbw1 = (eta * ((rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut).T.dot(dw2).T * r_Hidden).flatten('F')
                        gradient = np.gradient(dbw1, axis=0)
                        gradient = np.array(gradient, dtype=float)
                        grad_squared = 0.9 * grad_squared + 0.1
                        dbw1 = dbw1 - (eta/np.sqrt(grad_squared)) * gradient
                        bias_W1 += dbw1
                        
                        # Update variables for q-learning
                        QvalueOld = np.max(Q[allowed_a])
                        rectOut = output
                        hiddenOld = nOut1
                        inputOld = nX
                        rOld = R

                else:

                    """ SARSA """
    
                    # Rectified output
                    output = np.zeros((n_output_layer,1))
                    output[a_agent,0] = 1
                    
                    # Weighted updated for each iteration
                    dw2 = eta * (rOld - QvalueOld + gamma * Q[a_agent]) * rectOut.dot(hiddenOld.T)
                    W2 += dw2
                    
                    dbw2 = eta * (rOld - QvalueOld + gamma * Q[a_agent]) * rectOut.flatten('F')
                    bias_W2 += dbw2
                    
                    # Heaviside function
                    r_Hidden = np.heaviside(hiddenOld, 0)
                    
                    dw1 = eta * (((rOld - QvalueOld + gamma * Q[a_agent]) * rectOut).T.dot(dw2).T * r_Hidden).dot(inputOld.T)
                    W1 += dw1
                    
                    dbw1 = (eta * ((rOld - QvalueOld + gamma * Q[a_agent]) * rectOut).T.dot(dw2).T * r_Hidden).flatten('F')
                    bias_W1 += dbw1
                    
                    
                    # Update variables for sarsa
                    QvalueOld = Q[a_agent]
                    rectOut = output
                    hiddenOld = nOut1
                    inputOld = nX
                    rOld = R

                R_save[n,0] = rOld

                if n == 0:
                    S_ema[n,0] = 0.1
                    N_moves_ema[n,0] = 25
                S_ema[n+1,0] = (alpha * rOld) + (1-alpha)*S_ema[n,0]
                N_moves_ema[n+1,0] = (alpha * N_moves_save[n,0]) + (1-alpha)*N_moves_ema[n,0]
                # THE CODE ENDS HERE

                if checkmate:
                    break

            elif np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 0:
                # King 2 has no freedom but it is not checked
                draw = 1
                R = 0.1

                """
                Update the parameters of network by applying backpropagation and Q-learning. Use the 
                rectified linear function as activation function. Exploit the Q value for 
                the action made. This is the last iteration of the episode, it is a draw.
                """
                
                if alg_type == 0:
                
                    
                    """ Q-Learning """
                    if rms_use == 0:
                        #Rectified output 
                        output = np.zeros((n_output_layer,1))
                        output[a_agent,0] = 1
                        
                        # Weighted updated for each iteration
                        dw2 = eta * (rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut.dot(hiddenOld.T)
                        W2 += dw2
                        
                        dbw2 = eta * (rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut.flatten('F')
                        bias_W2 += dbw2
                        
                        # Heaviside function
                        r_Hidden = np.heaviside(hiddenOld, 0)
                        
                        dw1 = eta * (((rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut).T.dot(dw2).T * r_Hidden).dot(inputOld.T)
                        W1 += dw1
                        
                        dbw1 = (eta * ((rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut).T.dot(dw2).T * r_Hidden).flatten('F')
                        bias_W1 += dbw1
                        
                        # Update variables for q-learning
                        QvalueOld = np.max(Q[allowed_a])
                        rectOut = output
                        hiddenOld = nOut1
                        inputOld = nX
                        rOld = R
                        
                    else:
                        #Rectified output 
                        output = np.zeros((n_output_layer,1))
                        output[a_agent,0] = 1
                        
                        # Weighted updated for each iteration
                        dw2 = eta * (rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut.dot(hiddenOld.T)
                        gradient = np.gradient(dw2, axis=0)
                        gradient = np.array(gradient, dtype=float)
                        grad_squared = 0.9 * grad_squared + 0.1
                        dw2 = dw2 - (eta/np.sqrt(grad_squared)) * gradient
                        W2 += dw2
                        
                        dbw2 = eta * (rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut.flatten('F')
                        gradient = np.gradient(dbw2, axis=0)
                        gradient = np.array(gradient, dtype=float)
                        grad_squared = 0.9 * grad_squared + 0.1
                        dbw2 = dbw2 - (eta/np.sqrt(grad_squared)) * gradient
                        bias_W2 += dbw2
                        
                        # Heaviside function
                        r_Hidden = np.heaviside(hiddenOld, 0)
                        
                        dw1 = eta * (((rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut).T.dot(dw2).T * r_Hidden).dot(inputOld.T)
                        gradient = np.gradient(dw1, axis=0)
                        gradient = np.array(gradient, dtype=float)
                        grad_squared = 0.9 * grad_squared + 0.1
                        dw1 = dw1 - (eta/np.sqrt(grad_squared)) * gradient
                        W1 += dw1
                        
                        dbw1 = (eta * ((rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut).T.dot(dw2).T * r_Hidden).flatten('F')
                        gradient = np.gradient(dbw1, axis=0)
                        gradient = np.array(gradient, dtype=float)
                        grad_squared = 0.9 * grad_squared + 0.1
                        dbw1 = dbw1 - (eta/np.sqrt(grad_squared)) * gradient
                        bias_W1 += dbw1
                        
                        # Update variables for q-learning
                        QvalueOld = np.max(Q[allowed_a])
                        rectOut = output
                        hiddenOld = nOut1
                        inputOld = nX
                        rOld = R
                
                
                else: 
                
                    """ SARSA """
                    #Rectified output 
                    output = np.zeros((n_output_layer,1))
                    output[a_agent,0] = 1
                    
                    # Weighted updated for each iteration
                    dw2 = eta * (rOld - QvalueOld + gamma * Q[a_agent]) * rectOut.dot(hiddenOld.T)
                    W2 += dw2
                    
                    dbw2 = eta * (rOld - QvalueOld + gamma * Q[a_agent]) * rectOut.flatten('F')
                    bias_W2 += dbw2
                    
                    # Heaviside function
                    r_Hidden = np.heaviside(hiddenOld, 0)
                    
                    dw1 = eta * (((rOld - QvalueOld + gamma * Q[a_agent]) * rectOut).T.dot(dw2).T * r_Hidden).dot(inputOld.T)
                    W1 += dw1
                    
                    dbw1 = (eta * ((rOld - QvalueOld + gamma * Q[a_agent]) * rectOut).T.dot(dw2).T * r_Hidden).flatten('F')
                    bias_W1 += dbw1
                    
                    
                    # Update variables for sarsa
                    QvalueOld = Q[a_agent]
                    rectOut = output
                    hiddenOld = nOut1
                    inputOld = nX
                    rOld = R
                     
                R_save[n,0] = rOld
                if n == 0:
                    S_ema[n,0] = 0.1
                    N_moves_ema[n,0] = 25
                S_ema[n+1,0] = (alpha * rOld) + (1-alpha)*S_ema[n,0]
                N_moves_ema[n+1,0] = (alpha * N_moves_save[n,0]) + (1-alpha)*N_moves_ema[n,0]

                if draw:
                    break

            else:
                # Move enemy King randomly to a safe location
                allowed_enemy_a = np.where(a_k2 > 0)[0]
                a_help = int(np.ceil(np.random.rand() * allowed_enemy_a.shape[0]) - 1)
                a_enemy = allowed_enemy_a[a_help]

                direction = a_enemy
                steps = 1

                s[p_k2[0], p_k2[1]] = 0
                mov = map[direction, :] * steps
                s[p_k2[0] + mov[0], p_k2[1] + mov[1]] = 3

                p_k2[0] = p_k2[0] + mov[0]
                p_k2[1] = p_k2[1] + mov[1]

            # Update the parameters

            # Possible actions of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)
            # Compute features
            x_next = features(p_q1, p_k1, p_k2, dfK2, s, check)
            # Compute Q-values for the discounted factor
            Q_next, _ = Q_values(x_next, W1, W2, bias_W1, bias_W2)

            """
            Update the parameters of network by applying backpropagation and Q-learning. Use the 
            rectified linear function as activation function. 
            This is not the last iteration of the episode, the match continues.
            """


            if alg_type == 0:            
            
                """ Q-Learning """
                if rms_use == 0:
                    #Rectified output 
                    output = np.zeros((n_output_layer,1))
                    output[a_agent,0] = 1
                    
                    # Weighted updated for each iteration
                    dw2 = eta * (rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut.dot(hiddenOld.T) 
                    W2 += dw2
                    
                    dbw2 = eta * (rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut.flatten('F')
                    bias_W2 += dbw2
                    
                    # Heaviside function
                    r_Hidden = np.heaviside(hiddenOld, 0)
                    
                    dw1 = eta * (((rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut).T.dot(dw2).T * r_Hidden).dot(inputOld.T)
                    W1 += dw1
                    
                    dbw1 = (eta * ((rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut).T.dot(dw2).T * r_Hidden).flatten('F')
                    bias_W1 += dbw1
                    
                    # Update variables for q-learning
                    QvalueOld = np.max(Q[allowed_a])
                    rectOut = output
                    hiddenOld = nOut1
                    inputOld = nX
                    rOld = R
                else:
                    #Rectified output 
                    output = np.zeros((n_output_layer,1))
                    output[a_agent,0] = 1
                    
                    # Weighted updated for each iteration
                    dw2 = eta * (rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut.dot(hiddenOld.T) 
                    gradient = np.gradient(dw2, axis=0)
                    gradient = np.array(gradient, dtype=float)
                    grad_squared = 0.9 * grad_squared + 0.1
                    dw2 = dw2 - (eta/np.sqrt(grad_squared)) * gradient
                    W2 += dw2
                    
                    
                    dbw2 = eta * (rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut.flatten('F')
                    gradient = np.gradient(dbw2, axis=0)
                    gradient = np.array(gradient, dtype=float)
                    grad_squared = 0.9 * grad_squared + 0.1
                    dbw2 = dbw2 - (eta/np.sqrt(grad_squared)) * gradient
                    bias_W2 += dbw2
    
                    # Heaviside function
                    r_Hidden = np.heaviside(hiddenOld, 0)
                    
                    dw1 = eta * (((rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut).T.dot(dw2).T * r_Hidden).dot(inputOld.T)
                    gradient = np.gradient(dw1, axis=0)
                    gradient = np.array(gradient, dtype=float)
                    grad_squared = 0.9 * grad_squared + 0.1
                    dw1 = dw1 - (eta/np.sqrt(grad_squared)) * gradient
                    W1 += dw1
                    
                    dbw1 = (eta * ((rOld - QvalueOld + gamma * np.max(Q[allowed_a])) * rectOut).T.dot(dw2).T * r_Hidden).flatten('F')
                    gradient = np.gradient(dbw1, axis=0)
                    gradient = np.array(gradient, dtype=float)
                    grad_squared = 0.9 * grad_squared + 0.1
                    dbw1 = dbw1 - (eta/np.sqrt(grad_squared)) * gradient
                    bias_W1 += dbw1
                    
                    # Update variables for q-learning
                    QvalueOld = np.max(Q[allowed_a])
                    rectOut = output
                    hiddenOld = nOut1
                    inputOld = nX
                    rOld = R
            
            else:
            
                """ SARSA """
                
                #Rectified output 
                output = np.zeros((n_output_layer,1))
                output[a_agent,0] = 1
                
                # Weighted updated for each iteration
                dw2 = eta * (rOld - QvalueOld + gamma * Q[a_agent]) * rectOut.dot(hiddenOld.T)
                W2 += dw2
                
                dbw2 = eta * (rOld - QvalueOld + gamma * Q[a_agent]) * rectOut.flatten('F')
                bias_W2 += dbw2
                
                # Heaviside function
                r_Hidden = np.heaviside(hiddenOld, 0)
                
                dw1 = eta * (((rOld - QvalueOld + gamma * Q[a_agent]) * rectOut).T.dot(dw2).T * r_Hidden).dot(inputOld.T)
                W1 += dw1
                
                dbw1 = (eta * ((rOld - QvalueOld + gamma * Q[a_agent]) * rectOut).T.dot(dw2).T * r_Hidden).flatten('F')
                bias_W1 += dbw1
    
                
                # Update variables for sarsa
                QvalueOld = Q[a_agent]
                rectOut = output
                hiddenOld = nOut1
                inputOld = nX
                rOld = R
            
            N_moves_save[n,0] += 1
            R_save[n,0] = rOld
            if n == 0:
                S_ema[n,0] = 0.1
            S_ema[n+1,0] = (alpha * rOld) + (1-alpha)*S_ema[n,0]

            # YOUR CODE ENDS HERE
            i += 1

    return S_ema, N_moves_ema

if __name__ == '__main__': 
    
    episodes = 100000
    S_ema, N_moves_ema = main(0.00005, 0.05, episodes, 1, 0)
    S_ema2, N_moves_ema2 = main(0.000005, 0.0005, episodes, 1, 0)   
    nTrials = np.arange(episodes +1)
     
    def millions(x):
        return '$%1.1fM' % (x*1e-6)
    
    """ Rewards """
    fig, ax = plt.subplots()
    ax.fmt_ydata = millions
    ax.set_ylabel('Average Rewards')
    ax.set_xlabel('Episodes')
    plt.plot(nTrials,S_ema[:,0], label='Beta = 0.0005, Gamma = 0.05')
    plt.plot(nTrials,S_ema2[:,0], label='Beta = 0.00005, Gamma = 0.0005')
    plt.legend(loc='upper left', frameon=False)
    plt.show()
    
    """ Moves """
    fig, ax = plt.subplots()
    ax.fmt_ydata = millions
    ax.set_ylabel('Average Moves')
    ax.set_xlabel('Episodes')
    
    """ Plot results """
    plt.plot(nTrials,N_moves_ema[:,0], label='Beta = 0.0005, Gamma = 0.05')
    plt.plot(nTrials,N_moves_ema2[:,0], label='Beta = 0.00005, Gamma = 0.0005')
    plt.legend(loc='upper left', frameon=False)
    plt.show()
    

    