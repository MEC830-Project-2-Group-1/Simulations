import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from collections import deque

# System Constants
l = 40E-2  # pendulum length in meters
Mp = 50E-3  # Pendulum mass kg
Mc = 100E-3  # cart mass kg
D = 0.001  # damping coefficient of pendulum-cart joint
J = (Mp * l**2) /3  # Moment of inertia of pendulum

g = 9.81  # Earths gravity
k_air_pend = 0.0  # Air resistance constant for pendulum total speed chosen arbitrarily
k_air_cart=00.0; #ir resistance of square cart
dt = 0.01  # Essential frame rate of the simulation

# ID controller (found via trial and error)
Kp, Ki, Kd = 3, 00.00 , 0.1
#global integral, error_old
integral = 0
error_old = 0

global F_control_list
F_control_list= []

global theta_ddot

global theta_filtered
theta_filtered=0
theta_history = deque([np.radians(5)] * 10, maxlen=10)

# Equations of motion
def EOM(t, y, F_control):
    
    # Unpack current state from y
    theta, theta_dot, x, x_dot = y

    #USER INPUTS for time and force (direction implicit with sign)!
    t_imp_cart=1
    t_imp_pen=3

    #impulses
    if (t_imp_cart<t<t_imp_cart+0.1):
        F_imp_cart= 2
    else:
        F_imp_cart=0

    if (t_imp_pen<t<t_imp_pen+0.1):
        F_imp_pen=0
    else:
        F_imp_pen=0
    
    # air resistance at the pendulum's end
    F_air_angular = k_air_pend * (l * theta_dot)
    F_air_linear= k_air_pend*x_dot

    # air resistance at cart
    F_air_cart= k_air_cart*x_dot

    # Cart acceleration without theta ddot
    x_ddot_old = (F_imp_cart - F_air_cart + theta_dot**2 * l * Mp* D*np.sin(theta)) / Mc
    theta_ddot = (F_imp_pen*l + F_control*l+ Mp*g*np.sin(theta) + F_air_linear*l*np.sin(theta) - F_air_angular*l - D*theta_dot + Mp*x_ddot_old*l*np.cos(theta))/J

    #Friction
    N=Mc*9.81+(Mp*l*theta_dot**2 *np.cos(theta)) + (Mp*l*theta_ddot*np.sin(theta))

    if abs(x_dot) < 1:
        direction=0;
    else :
         direction=x_dot/abs(x_dot) #Friction always apposes the direction of motion but is not proportional to it; used to determine sign of friction term

    F_friction = 0.5 * N * direction

    x_ddot = (F_imp_cart - F_friction + F_control - F_air_cart + theta_dot**2 * l * Mp* np.sin(theta) - theta_ddot*l*Mp*np.cos(theta)) / Mc
    theta_ddot = (F_imp_pen*l + F_control*l + Mp*g*np.sin(theta) + F_air_linear*l*np.sin(theta) - F_air_angular*l - D*theta_dot + Mp*x_ddot_old*l*np.cos(theta))/J
    
    #send variables back
    return [theta_dot, theta_ddot, x_dot, x_ddot]

# PID controller for system
def PID(goal_angle, current_angle,t):
    global integral, error_old, theta_filtered, F_control_previous, theta_history
    percent_keep=0.3

    if t==0:
        F_control_previous=0
    
    #add noise to signal
    noise_mean = 0 #random noise therefore average is zero
    noise_sigma_squared = 0.5  # Adjust this value for desired noise level
    theta_noise = current_angle + np.random.normal(noise_mean, noise_sigma_squared)  # Gaussian noise factor of scaling of one chosen arbitrarily

    #since we are dealing with essentially pure random error taking the average of the meassuremnt was seen to be the best opton
    #theta_filtered = (percent_keep * theta_filtered) + ((1 - percent_keep) * theta_noise) 
    theta_history.append(theta_noise)

    if len(theta_history) > 0:
        theta_average = sum(theta_history) / len(theta_history)
    else:
        theta_average = current_angle

    #noise signal not used as my computer isnt strong enough to load the program if used. Calculated for demonstrated activities
    error = goal_angle - current_angle #use theta_filtered if your computer can handle it
    integral = integral + error  # Area under curve with rectangular approximation of width dt
    integral = np.clip(integral, -10, 10) # preventing unbounded growth
    
    if error==0:
        integral=0 #ressetting area term once we get zero area

    derivative = (error - error_old) / dt  # Change in error over change in time
    
    
    
    F_control=F_control_previous + 100*dt*np.clip(Kp * error + Ki * integral + Kd * derivative, -10, 10) #total control signal assuming maximum possible force is 10N
    #F_control=np.clip(Kp * error + Ki * integral + Kd * derivative, -10, 10)

    print(F_control)
    error_old = error #passing old error for next loop
    
    return F_control

# Run simulation
def main():

    # Initial conditions: [theta, theta_dot, x, x_dot]
    y = [np.radians(5), 0, 0, 0]  #array reresents the state of the system

    # Time parameters
    time_span = (0, 10)  # Simulate for 30 seconds
    # Evaluation points
    t_eval=np.linspace(0, 10, 200)

    # Calculating state values with equation of motion
    def ode_system(t, y):
        
        F_control = PID(0, y[0],t)  # Target angle is zero (vertical y axis)
        F_control_list.append(F_control)
        return EOM(t, y, F_control)

    # Solving ODEs
    sol = solve_ivp(ode_system, time_span, y, t_eval=t_eval) #solves ODE's w/ DE's for 10 seconds using IC's evaluated at times t_eval
    
    # Visualize stuff
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 12)) #initoalize sub plots
    
    for i in range(len(t_eval)+1): #loops for every solution of "sol" (0-10 seconds w/ 300 points in between) +1 to display all 300 points
        theta, theta_dot, x, x_dot = sol.y[:, i] #storing state variables in 2d matrix, each one with their own row each column is a value at t_eval
        
        # Update pendulum and cart positions
        cart_pos = x #positon of cart and pendulum R joint
        xp = x + l * np.sin(theta) #x component of pendlum point mass location
        yp = l * np.cos(theta) # y component
        
        # Clear previous plots after every iteration
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
        
        # Draw cart on display 1
        ax1.plot([cart_pos - 1, cart_pos + 1], [0, 0], 'k', lw=5)
        
        # Draw pendulum on display 1
        ax1.plot([cart_pos, xp], [0, yp], 'r', lw=2)
        ax1.plot(xp, yp, 'bo', markersize=10)
        
        # Set plot cart position
        ax1.set_xlim([-20, 20])
        ax1.set_ylim([-0.5, 0.5]) #note: becasue of the extreme difference in scale in the x and y
        ax1.set_title("Inverted Pendulum-Cart System: PID controller")
        ax1.set_xlabel("Location (m)")
        ax1.set_ylabel("Height (m)")
        
        # Plot theta over time
        ax2.plot(sol.t[:i + 1], np.degrees(sol.y[0, :i + 1]), 'b-', label= "Theta (rad)")
        ax2.set_xlim(time_span)
        ax2.set_ylim([-180, 180])
        ax2.set_xlabel("Time(s)")
        ax2.set_ylabel("Pendulum Angle deg")
        ax2.legend()
        
        # Plot cart position over time
        ax3.plot(sol.t[:i + 1], sol.y[2, :i + 1], 'g-', label= "Cart Position (m)")
        ax3.set_xlim(time_span)
        ax3.set_ylim([-60, 60])
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Cart Position (m)")
        ax3.legend()

        #plot control force vs time
        #ax4.plot(sol.t[:i + 1], F_control_list[:i + 1], 'r-', label= "Control Force (N)")
        ax4.plot(t_eval[:i + 1], F_control_list[:i + 1], 'r-', label="Control Force (N)")
        ax4.set_xlim(time_span)
        ax4.set_ylim([-1, 1])  # Adjust the limits as needed
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Control Force (N)")
        ax4.legend()
        
        plt.pause(dt)  # Pause to allow for animation

    plt.show()

# Run the simulation
main()
