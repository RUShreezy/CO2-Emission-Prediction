import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# To run: streamlit run app.py

# Load the saved model
model_path = 'ML_MODEL\\random_forest_model.pkl'
model = joblib.load(model_path)

# Extract feature names from the model if they are available
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("The model does not contain feature names. Please ensure the model is trained with feature names.")

def main():
    # Set the title of the web app
    st.title('Vehicle CO2 Level Prediction')

    # Add a description
    st.write('Enter Vehicle information to predict High or Low CO2 Levels.')

    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Vehicle Information')

        # Add input fields for features
        Vehicle_Class = st.selectbox("Vehicle Class", ['Compact', 'Two-seater', 'SUV: Small', 'Mid-size', 'Minicompact', 'SUV: Standard', 'Station wagon: Small', 'Subcompact', 'Station wagon: Mid-size', 'Full-size', 'Pickup truck: Small', 'Pickup truck: Standard', 'Minivan', 'Van: Passenger', 'Special purpose vehicle'])
        Engine_Size  = st.slider("Vehicle Engine Size", 1, 8, 4)
        Cylinders  = st.slider("Vehicle Cylinder", 3, 16, 4)
        Transmission = st.selectbox("Vehicle Transmission", ['AM8', 'AM9', 'AS10', 'A8', 'A9', 'M7', 'AM7', 'AS8', 'M6', 'AS6', 'AV', 'AS9', 'A10', 'A6', 'M5', 'AV7', 'AV1', 'AM6', 'AS7', 'AV8', 'AV6', 'AV10', 'AS5'])
        Fuel_Consumption_in_City  = st.slider("Fuel Consumption in Offroad (L/100 km)", 3, 15, 10)
        Fuel_Consumption_in_City_Hwy  = st.slider("Fuel Consumption in City Highway (L/100 km)", 3, 20, 10)
        Fuel_Consumption_comb  = st.slider("Fuel Consumption in Normal Traffic (L/100 km)", 3, 20, 10)


    # Convert categorical inputs to numerical
    vehicle_class_mapping = {
        'Compact': 0,
        'Two-seater': 1,
        'SUV: Small': 2,
        'Mid-size': 3,
        'Minicompact': 4,
        'SUV: Standard': 5,
        'Station wagon: Small': 6,
        'Subcompact': 7,
        'Station wagon: Mid-size': 8,
        'Full-size': 9,
        'Pickup truck: Small': 10,
        'Pickup truck: Standard': 11,
        'Minivan': 12,
        'Van: Passenger': 13,
        'Special purpose vehicle': 14
    }
    transmission_mapping = {
        'AM8': 0,
        'AM9': 1,
        'AS10': 2,
        'A8': 3,
        'A9': 4,
        'M7': 5,
        'AM7': 6,
        'AS8': 7,
        'M6': 8,
        'AS6': 9,
        'AV': 10,
        'AS9': 11,
        'A10': 12,
        'A6': 13,
        'M5': 14,
        'AV7': 15,
        'AV1': 16,
        'AM6': 17,
        'AS7': 18,
        'AV8': 19,
        'AV6': 20,
        'AV10': 21,
        'AS5': 22
    }

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Vehicle_Class_Compact': [1 if vehicle_class_mapping == 0 else 0],
        'Vehicle_Class_Two-seater': [1 if vehicle_class_mapping == 1 else 0],
        'Vehicle_Class_SUV: Small': [1 if vehicle_class_mapping == 2 else 0],
        'Vehicle_Class_Mid-size': [1 if vehicle_class_mapping == 3 else 0],
        'Vehicle_Class_Minicompact': [1 if vehicle_class_mapping == 4 else 0],
        'Vehicle_Class_SUV: Standard': [1 if vehicle_class_mapping == 5 else 0],
        'Vehicle_Class_Station wagon: Small': [1 if vehicle_class_mapping == 6 else 0],
        'Vehicle_Class_Subcompact': [1 if vehicle_class_mapping == 7 else 0],
        'Vehicle_Class_Station wagon: Mid-size': [1 if vehicle_class_mapping == 8 else 0],
        'Vehicle_Class_Full-size': [1 if vehicle_class_mapping == 9 else 0],
        'Vehicle_Class_Pickup truck: Small': [1 if vehicle_class_mapping == 10 else 0],
        'Vehicle_Class_Pickup truck: Standard': [1 if vehicle_class_mapping == 11 else 0],
        'Vehicle_Class_Minivan': [1 if vehicle_class_mapping == 12 else 0],
        'Vehicle_Class_Van: Passenger': [1 if vehicle_class_mapping == 13 else 0],
        'Vehicle_Class_Special purpose vehicle': [1 if vehicle_class_mapping == 14 else 0],
        'Transmission_AM8': [1 if transmission_mapping == 0 else 0],
        'Transmission_AM9': [1 if transmission_mapping == 1 else 0],
        'Transmission_AS10': [1 if transmission_mapping == 2 else 0],
        'Transmission_A8': [1 if transmission_mapping == 3 else 0],
        'Transmission_A9': [1 if transmission_mapping == 4 else 0],
        'Transmission_M7': [1 if transmission_mapping == 5 else 0],
        'Transmission_AM7': [1 if transmission_mapping == 6 else 0],
        'Transmission_AS8': [1 if transmission_mapping == 7 else 0],
        'Transmission_M6': [1 if transmission_mapping == 8 else 0],
        'Transmission_AS6': [1 if transmission_mapping == 9 else 0],
        'Transmission_AV': [1 if transmission_mapping == 10 else 0],
        'Transmission_AS9': [1 if transmission_mapping == 11 else 0],
        'Transmission_A10': [1 if transmission_mapping == 12 else 0],
        'Transmission_A6': [1 if transmission_mapping == 13 else 0],
        'Transmission_M5': [1 if transmission_mapping == 14 else 0],
        'Transmission_AV7': [1 if transmission_mapping == 15 else 0],
        'Transmission_AV1': [1 if transmission_mapping == 16 else 0],
        'Transmission_AM6': [1 if transmission_mapping == 17 else 0],
        'Transmission_AS7': [1 if transmission_mapping == 18 else 0],
        'Transmission_AV8': [1 if transmission_mapping == 19 else 0],
        'Transmission_AV6': [1 if transmission_mapping == 20 else 0],
        'Transmission_AV10': [1 if transmission_mapping == 21 else 0],
        'Transmission_AS5': [1 if transmission_mapping == 22 else 0],
        'Engine_Size': [Engine_Size],
        'Cylinders': [Cylinders],
        'Fuel_Consumption_in_City': [Fuel_Consumption_in_City],
        'Fuel_Consumption_in_City_Hwy': [Fuel_Consumption_in_City_Hwy],
        'Fuel_Consumption_comb': [Fuel_Consumption_comb]
    })

    # Ensure columns are in the same order as during model training
    input_data = input_data[expected_columns]

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]
            
            st.write(f'Prediction: {"High CO2 Emission Level" if prediction[0] == 1 else "Normal CO2 Emission Level"}')
            st.write(f'Probability of CO2 Emission Level: {probability:.2f}')

            # Plotting
            fig, axes = plt.subplots(2, 1, figsize=(8, 16))

            # Plot High/Normal probability
            sns.barplot(x=['Normal', 'High'], y=[1 - probability, probability], ax=axes[0], palette=['green', 'red'])
            axes[0].set_title('High/Normal Probability')
            axes[0].set_ylabel('Probability')

            # Plot High/Normal pie chart
            axes[1].pie([1 - probability, probability], labels=['Normal', 'High'], autopct='%1.1f%%', colors=['green', 'red'])
            axes[1].set_title('High/Normal Pie Chart')

            # Display the plots
            st.pyplot(fig)

            # Provide recommendations
            if prediction[0] == 1:
                st.error("Warning! The Vehicle is likely to emmit high level of CO2 emission!")
            else:
                st.success("The Vehicle is likely to emmit normal level of CO2 emission. Good!")

if __name__ == '__main__':
    main()
