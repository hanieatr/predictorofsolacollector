import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from PIL import Image
import time

user_choice = st.radio("please choose the function of prediction ",
                       ("feature-selection input datas", "all input datas"))
#......................
if user_choice == "all input datas":
   # Title of the app
    st.title("Solar Thermal Collector Performance Prediction")

# Load and display image
    photo = Image.open('gui2.png')
    st.image(photo)
    
    
# Dictionary to map fluid names to saturation temperatures
    fluid_temps = {
    'Acetone': 0.000000169,
    'Methanol':9.23E-08,
    'Ethanol': 7.13E-08,
    'Water': 0.000000169
    }
# Sidebar inputs
    manifold_fluid = st.sidebar.radio('Manifold fluid', ['Water', 'Air'])
#using dct
    fluid_name = st.sidebar.radio('Heatpipe Fluid', list(fluid_temps.keys()))
    alpha = fluid_temps[fluid_name] # Get saturation temperature from dictionary

    N = st.sidebar.number_input('Number of tubes (N)', min_value=1)
    D_o = st.sidebar.number_input('Outer diameter of tube (D_o)',format="%.4f")
    L_tube = st.sidebar.number_input('Length of the tube (L_tube)',format="%.4f")
    L_c = st.sidebar.number_input('Length of the condenser (L_c)',format="%.4f")
    L_e = st.sidebar.number_input('Length of the evaporator (L_e)',format="%.4f")
    L_a = st.sidebar.number_input('Length of the adiabatic section (L_a)',format="%.4f")
    De_o = st.sidebar.number_input('Outer diameter of evaporator (De_o)',format="%.4f")
    t_e = st.sidebar.number_input('Thickness of evaporator (t_e)',format="%.4f")
    Dc_o = st.sidebar.number_input('Outer diameter of condenser (Dc_o)',format="%.4f")
    t_c = st.sidebar.number_input('Thickness of condenser (t_c)',format="%.4f")
    theta = st.sidebar.number_input('Angle of radiation (theta)',format="%.4f")
    t_g = st.sidebar.number_input('Thickness of glass (t_g)',format="%.4f")
    D_H = st.sidebar.number_input('Hydraulic diameter of tube (D_H)',format="%.4f")
    A_man = st.sidebar.number_input('Area of manifold (A_man)',format="%.4f")
    alpha_ab = st.sidebar.number_input('Absorptivity of absorber (alpha_ab)',format="%.4f")
    epsilon_ab = st.sidebar.number_input('Emissivity of absorber (epsilon_ab)',format="%.4f")
    epsilon_g = st.sidebar.number_input('Emissivity of glass (epsilon_g)',format="%.4f")
    tau_g = st.sidebar.number_input('Transmissivity of glass (tau_g)',format="%.4f")
    I = st.sidebar.number_input('Solar irradiance (I)',format="%.4f")
    T_amb = st.sidebar.number_input('Ambient temperature (T_amb)',format="%.4f")
    U_amb = st.sidebar.number_input('Velocity of wind (U_amb)',format="%.4f")
    T_in = st.sidebar.number_input('Inlet temperature (T_in)',format="%.4f")
    m_dot = st.sidebar.number_input('Mass flow rate of manifold fluid (m_dot)',format="%.4f")

# Prepare input data for prediction
    input_data = pd.DataFrame({
    'alpha': [alpha],
    'N': [N],
    'D_o': [D_o],
    'L_tube': [L_tube],
    'L_c': [L_c],
    'L_e': [L_e],
    'L_a': [L_a],
    'De_o': [De_o],
    't_e': [t_e],
    'Dc_o': [Dc_o],
    't_c': [t_c],
    'theta': [theta],
    't_g': [t_g],
    'D_H': [D_H],
    'A_man': [A_man],
    'alpha_ab': [alpha_ab],
    'epsilon_ab': [epsilon_ab],
    'epsilon_g': [epsilon_g],
    'tau_g': [tau_g],
    'I': [I],
    'T_amb': [T_amb],
    'U_amb': [U_amb],
    'T_in': [T_in],
    'm_dot': [m_dot]
    })
#.................create process bar
    progress_bar=st.progress(0)

# Load data based on fluid type selection
    if manifold_fluid == 'Water':
    # Load models and scalers for Water
        with open('ext_allinput_water_eta', 'rb') as f:
            loaded_model_eta = pickle.load(f)
        with open('ext_allinput_water_T', 'rb') as f:
            loaded_model_T = pickle.load(f)

#....................................................
        with open('Xwatereta_sc_extr.pkl', 'rb') as f:
            x_scaler_eta = pickle.load(f)
        with open('Ywatereta_sc_extr.pkl', 'rb') as f:
            scaler_y_eta = pickle.load(f)

        with open('XwatereT_sc_extr.pkl', 'rb') as f:
            x_scaler_T = pickle.load(f)
        with open('YwaterT_sc_extr.pkl', 'rb') as f:
            scaler_y_T = pickle.load(f)
        #............................
        progress_bar.progress(33)        #1/3 of complication

    # Make predictions for both models
        input_data_scaled_eta = x_scaler_eta.transform(input_data.values.reshape(1, -1))
        y_pred_scaled_eta = loaded_model_eta.predict(input_data_scaled_eta)
        y_pred_eta = scaler_y_eta.inverse_transform(y_pred_scaled_eta.reshape(-1, 1))

        input_data_scaled_T = x_scaler_T.transform(input_data.values.reshape(1, -1))
        y_pred_scaled_T = loaded_model_T.predict(input_data_scaled_T)
        y_pred_T = scaler_y_T.inverse_transform(y_pred_scaled_T.reshape(-1, 1))
#..................
        progress_bar.progress(66)
    
    # Display prediction results
        st.subheader("Predicted Outputs:")
        st.write("Predicted Efficiency (eta):", y_pred_eta[0][0],format="%.4f")
        st.write("Predicted Exit Temperature (T):", y_pred_T[0][0],format="%.4f") 

        progress_bar.progress(100)
    else: # manifold_fluid == 'Air'
        # Load models and scalers for air
        with open('ext_allinput_air_eta', 'rb') as f:
            loaded_model_eta = pickle.load(f)
        with open('ext_allinput_air_T', 'rb') as f:
            loaded_model_T = pickle.load(f)

#....................................................
        with open('Xaireta_sc_extr.pkl', 'rb') as f:
            x_scaler_eta = pickle.load(f)
        with open('Yairreta_sc_extr.pkl', 'rb') as f:
            scaler_y_eta = pickle.load(f)

        with open('XairT_sc_extr.pkl', 'rb') as f:
            x_scaler_T = pickle.load(f)
        with open('YairT_sc_extr.pkl', 'rb') as f:
            scaler_y_T = pickle.load(f)
        #..............
        progress_bar.progress(33) 

    # Make predictions for both models
        input_data_scaled_eta = x_scaler_eta.transform(input_data.values.reshape(1, -1))
        y_pred_scaled_eta = loaded_model_eta.predict(input_data_scaled_eta)
        y_pred_eta = scaler_y_eta.inverse_transform(y_pred_scaled_eta.reshape(-1, 1))

        input_data_scaled_T = x_scaler_T.transform(input_data.values.reshape(1, -1))
        y_pred_scaled_T = loaded_model_T.predict(input_data_scaled_T)
        y_pred_T = scaler_y_T.inverse_transform(y_pred_scaled_T.reshape(-1, 1))
    #......
        progress_bar.progress(66) 

    # Display prediction results
        st.subheader("Predicted Outputs for Water:")
        st.write("Predicted Efficiency (eta):", y_pred_eta[0][0],format="%.4f")
        st.write("Predicted Exit Temperature (T_exit):", y_pred_T[0][0],format="%.4f")
        progress_bar.progress(100) 
        
        #########################################################################..................
        ##########################33
        ###########################33
        #.....................................feature selection
        #.................................................
        # Load data based on fluid type selection
else: #*******************second back end featuuureeeeeeeee
      # Title of the app
    st.title("Solar Thermal Collector Performance Prediction")

# Load and display image
    photo = Image.open('gui2.png')
    st.image(photo)
    manifold_fluid = st.sidebar.radio('Manifold fluid', ['Water', 'Air'])
    predicted_parameter = st.sidebar.radio('predicted_parameter', ['Temperature', 'efficiency'])
    
    if manifold_fluid == 'Water':
        if predicted_parameter=='Temperature':
            
            fluid_temps = {
            'Acetone': 0.000000169,
            'Methanol':9.23E-08,
            'Ethanol': 7.13E-08,
            'Water': 0.000000169
            }
            fluid_name = st.sidebar.radio('Heatpipe Fluid', list(fluid_temps.keys()))
            alpha = fluid_temps[fluid_name] # Get saturation temperature from dictionary
            m_dot = st.sidebar.number_input('Mass flow rate of manifold fluid (m_dot)',format="%.4f")
            N = st.sidebar.number_input('Number of tubes (N)', min_value=1,format="%.4f")
            I = st.sidebar.number_input('Solar irradiance (I)',format="%.4f")
            T_in = st.sidebar.number_input('Inlet temperature (T_in)',format="%.4f")
            T_amb = st.sidebar.number_input('Ambient temperature (T_amb)',format="%.4f")
             # Dictionary to map fluid names to saturation temperatures
            
            input_data_T_water = pd.DataFrame({
            'alpha': [alpha],
            'T_in': [T_in],
            'm_dot': [m_dot],
            'N': [N],
            'I': [I],
            'T_amb': [T_amb],})
            progress_bar=st.progress(0)
    
            with open('ext_F_water_T', 'rb') as f:
                loaded_model_T = pickle.load(f)

#....................................................
            with open('XwaterF_T_sc_extr.pkl', 'rb') as f:
                x_scaler_T = pickle.load(f)
            with open('YwaterF_T_sc_extr.pkl', 'rb') as f:
                scaler_y_T = pickle.load(f)
        #............................
            progress_bar.progress(33)        #1/3 of complication

    # Make predictions for both models

            input_data_scaled_T = x_scaler_T.transform(input_data_T_water.values.reshape(1, -1))
            y_pred_scaled_T = loaded_model_T.predict(input_data_scaled_T)
            y_pred_T = scaler_y_T.inverse_transform(y_pred_scaled_T.reshape(-1, 1))
#..................
            progress_bar.progress(66)
    
    # Display prediction results
            st.subheader("Predicted Outputs:")
            st.write("Predicted outlet Temperature (T_exit):", y_pred_T[0],format="%.4f")  
            progress_bar.progress(100)
   #.........................

        else:#target is water efficiency
    # Dictionary to map fluid names to saturation temperatures

            fluid_temps = {
            'Acetone': 0.000000169,
            'Methanol':9.23E-08,
            'Ethanol': 7.13E-08,
            'Water': 0.000000169
            }
            fluid_name = st.sidebar.radio('Heatpipe Fluid', list(fluid_temps.keys()))
            alpha = fluid_temps[fluid_name] # Get saturation temperature from dictionary 
            tau_g = st.sidebar.number_input('Transmissivity of glass (tau_g)',format="%.4f")
            I = st.sidebar.number_input('Solar irradiance (I)',format="%.4f")
            T_in = st.sidebar.number_input('Inlet temperature (T_in)',format="%.4f")
            alpha_ab = st.sidebar.number_input('Absorptivity of absorber (alpha_ab)',format="%.4f")
            m_dot = st.sidebar.number_input('Mass flow rate of manifold fluid (m_dot)',format="%.4f")
            T_amb = st.sidebar.number_input('Ambient temperature (T_amb)',format="%.4f")
            L_c = st.sidebar.number_input('Length of the condenser (L_c)',format="%.4f")
            theta = st.sidebar.number_input('Angle of radiation (theta)',format="%.4f")
            N = st.sidebar.number_input('Number of tubes (N)', min_value=1,format="%.4f")
            D_o = st.sidebar.number_input('Outer diameter of tube (D_o)',format="%.4f")
              
            input_data_ETA_water = pd.DataFrame({
            'alpha': [alpha],   
            'tau_g': [tau_g],
            'I': [I],
            'T_in': [T_in],
            'alpha_ab': [alpha_ab],
            'm_dot': [m_dot],
            'T_amb': [T_amb],
            'L_c': [L_c],  
            'theta': [theta],
            'N': [N],
            'D_o': [D_o]
            })
            progress_bar=st.progress(0)
          
            with open('ext_F_water_eta', 'rb') as f:
                loaded_model_eta = pickle.load(f)

            with open('XwaterF_eta_sc_extr.pkl', 'rb') as f:
                x_scaler_eta = pickle.load(f)
            with open('YwaterF_eta_sc_extr.pkl', 'rb') as f:
                scaler_y_eta = pickle.load(f)
        #............................
            progress_bar.progress(33)        #1/3 of complication

    # Make predictions for both models

            input_data_scaled_eta = x_scaler_eta.transform(input_data_ETA_water.values.reshape(1, -1))
            y_pred_scaled_eta = loaded_model_eta.predict(input_data_scaled_eta)
            y_pred_eta = scaler_y_eta.inverse_transform(y_pred_scaled_eta.reshape(-1, 1))
#..................
            progress_bar.progress(66)
    
    # Display prediction results
            st.subheader("Predicted Outputs:")
            st.write("Predicted efficiency (Efficiency):", y_pred_eta[0],format="%.4f")  
            progress_bar.progress(100)
            
    else:  #manifold fluid is air&&&&&&&&&&&&&&&&&&&&&&&&&&&7%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        if predicted_parameter=='Temperature': 
            
            fluid_temps = {
            'Acetone': 0.000000169,
            'Methanol':9.23E-08,
            'Ethanol': 7.13E-08,
            'Water': 0.000000169
            }
            fluid_name = st.sidebar.radio('Heatpipe Fluid', list(fluid_temps.keys()))
            alpha = fluid_temps[fluid_name] # Get saturation temperature from dictionary 
            tau_g = st.sidebar.number_input('Transmissivity of glass (tau_g)',format="%.4f")
            I = st.sidebar.number_input('Solar irradiance (I)',format="%.4f")
            T_in = st.sidebar.number_input('Inlet temperature (T_in)',format="%.4f")
            N = st.sidebar.number_input('Number of tubes (N)', min_value=1,format="%.4f")
            m_dot = st.sidebar.number_input('Mass flow rate of manifold fluid (m_dot)',format="%.4f")
            D_o = st.sidebar.number_input('Outer diameter of tube (D_o)',format="%.4f")
            L_c = st.sidebar.number_input('Length of the condenser (L_c)',format="%.4f")
            epsilon_g = st.sidebar.number_input('Emissivity of glass (epsilon_g)',format="%.4f")
            theta = st.sidebar.number_input('Angle of radiation (theta)',format="%.4f")
            L_tube = st.sidebar.number_input('Length of the tube (L_tube)',format="%.4f")
            
            input_data_T_air = pd.DataFrame({
            'alpha': [alpha],  
            'm_dot': [m_dot],
            'N': [N],
            'I': [I],
            'T_in': [T_in],
            'D_o': [D_o],
            'theta': [theta],
            'L_c': [L_c],
            'L_tube': [L_tube],  
            'epsilon_g': [epsilon_g],
            'tau_g': [tau_g],
            })
            progress_bar=st.progress(0)
    
            with open('ext_F_air_T', 'rb') as f:
                loaded_model_T = pickle.load(f)

            with open('XairF_T_sc_extr.pkl', 'rb') as f:
                x_scaler_T = pickle.load(f)
            with open('YairF_T_sc_extr.pkl', 'rb') as f:
                scaler_y_T = pickle.load(f)
        #............................
            progress_bar.progress(33)        #1/3 of complication

    # Make predictions for both models

            input_data_scaled_T = x_scaler_T.transform(input_data_T_air.values.reshape(1, -1))
            y_pred_scaled_T = loaded_model_T.predict(input_data_scaled_T)
            y_pred_T = scaler_y_T.inverse_transform(y_pred_scaled_T.reshape(-1, 1))
#..................
            progress_bar.progress(66)
    
    # Display prediction results
            st.subheader("Predicted Outputs:")
            st.write("Predicted outlet Temperature (T_exit):", y_pred_T[0],format="%.4f")  
            progress_bar.progress(100)
   #.........................

        else:#target is air efficiency*********************
  
            fluid_temps = {
            'Acetone': 0.000000169,
            'Methanol':9.23E-08,
            'Ethanol': 7.13E-08,
            'Water': 0.000000169
            }
            fluid_name = st.sidebar.radio('Heatpipe Fluid', list(fluid_temps.keys()))
            alpha = fluid_temps[fluid_name] # Get saturation temperature from dictionary
            tau_g = st.sidebar.number_input('Transmissivity of glass (tau_g)',format="%.4f")
            I = st.sidebar.number_input('Solar irradiance (I)',format="%.4f")
            T_in = st.sidebar.number_input('Inlet temperature (T_in)',format="%.4f")
            epsilon_g = st.sidebar.number_input('Emissivity of glass (epsilon_g)',format="%.4f")
            m_dot = st.sidebar.number_input('Mass flow rate of manifold fluid (m_dot)',format="%.4f")
            L_tube = st.sidebar.number_input('Length of the tube (L_tube)',format="%.4f")
            L_c = st.sidebar.number_input('Length of the condenser (L_c)',format="%.4f")
            D_o = st.sidebar.number_input('Outer diameter of tube (D_o)',format="%.4f")
            D_H = st.sidebar.number_input('Hydraulic diameter of tube (D_H)',format="%.4f")
            N = st.sidebar.number_input('Number of tubes (N)', min_value=1,format="%.4f")
            
            
            input_data_ETA_air = pd.DataFrame({
            'alpha': [alpha],  
            'm_dot': [m_dot],
            'L_c': [L_c],
            'tau_g': [tau_g],
            'D_o': [D_o],
            'L_tube': [L_tube],
            'N': [N],
            'T_in': [T_in],
            'D_H': [D_H],  
            'epsilon_g': [epsilon_g],
            'I': [I],
            })
            progress_bar=st.progress(0)
           
            with open('ext_F_air_eta', 'rb') as f:
                loaded_model_eta = pickle.load(f)
            
            with open('XairF_eta_sc_extr.pkl', 'rb') as f:
                x_scaler_eta = pickle.load(f)
            with open('YairF_eta_sc_extr.pkl', 'rb') as f:
                scaler_y_eta = pickle.load(f)
        #............................
            progress_bar.progress(33)        #1/3 of complication

    # Make predictions for both models

            input_data_scaled_eta = x_scaler_eta.transform(input_data_ETA_air.values.reshape(1, -1))
            y_pred_scaled_eta = loaded_model_eta.predict(input_data_scaled_eta)
            y_pred_eta = scaler_y_eta.inverse_transform(y_pred_scaled_eta.reshape(-1, 1))
#..................
            progress_bar.progress(66)
    
    # Display prediction results
            st.subheader("Predicted Outputs:")
            st.write("Predicted efficiency (Efficiency):", y_pred_eta[0],format="%.4f")  
            progress_bar.progress(100)
        
        
        
        
        
        
        
        
        
        
        
        
  
        
      