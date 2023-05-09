import streamlit as st
import numpy as np
import joblib

from prediction import get_prediction, one_hot_encoder_building_class, label_encoder_State_Factor,  target_encoder_facility_type

model = joblib.load(r'models/SiteEnergyIntensityPrediction.joblib')

st.set_page_config(page_title="Site Energy Intensity Prediction App", page_icon="⚡", layout="wide")

Options_building_class  = ['Commercial', 'Residential']	
Options_State_Factor	 = ['State_1', 'State_2', 'State_4', 'State_6', 'State_8', 'State_10','State_11']	
Options_facility_type = ['Grocery_store_or_food_market',
                        'Warehouse_Distribution_or_Shipping_center',
                        'Retail_Enclosed_mall', 'Education_Other_classroom',
                        'Warehouse_Nonrefrigerated', 'Warehouse_Selfstorage',
                        'Office_Uncategorized', 'Data_Center', 'Commercial_Other',
                        'Mixed_Use_Predominantly_Commercial',
                        'Office_Medical_non_diagnostic', 'Education_College_or_university',
                        'Industrial', 'Laboratory',
                        'Public_Assembly_Entertainment_culture',
                        'Retail_Vehicle_dealership_showroom', 'Retail_Uncategorized',
                        'Lodging_Hotel', 'Retail_Strip_shopping_mall',
                        'Education_Uncategorized', 'Health_Care_Inpatient',
                        'Public_Assembly_Drama_theater', 'Public_Assembly_Social_meeting',
                        'Religious_worship', 'Mixed_Use_Commercial_and_Residential',
                        'Office_Bank_or_other_financial', 'Parking_Garage',
                        'Commercial_Unknown', 'Service_Vehicle_service_repair_shop',
                        'Service_Drycleaning_or_Laundry', 'Public_Assembly_Recreation',
                        'Service_Uncategorized', 'Warehouse_Refrigerated',
                        'Food_Service_Uncategorized', 'Health_Care_Uncategorized',
                        'Food_Service_Other', 'Public_Assembly_Movie_Theater',
                        'Food_Service_Restaurant_or_cafeteria', 'Food_Sales',
                        'Public_Assembly_Uncategorized', 'Nursing_Home',
                        'Health_Care_Outpatient_Clinic', 'Education_Preschool_or_daycare',
                        '5plus_Unit_Building', 'Multifamily_Uncategorized',
                        'Lodging_Dormitory_or_fraternity_sorority',
                        'Public_Assembly_Library', 'Public_Safety_Uncategorized',
                        'Public_Safety_Fire_or_police_station', 'Office_Mixed_use',
                        'Public_Assembly_Other', 'Public_Safety_Penitentiary',
                        'Health_Care_Outpatient_Uncategorized', 'Lodging_Other',
                        'Mixed_Use_Predominantly_Residential', 'Public_Safety_Courthouse',
                        'Public_Assembly_Stadium', 'Lodging_Uncategorized',
                        '2to4_Unit_Building', 'Warehouse_Uncategorized']	

#creating option list for dropdown menu
st.markdown("<h1 style='text-align: center;'>Site Energy Intensity Prediction App ⚡</h1>", unsafe_allow_html=True)
st.markdown("This project is part of my personal portfolio.")
st.markdown("In this, an attempt is made to predict Site EUI given the characteristics of the building.")
st.markdown("The target feature is **:red[Site EUI]** .")
st.markdown("The metric used for evaluation is **:green[RMSE]**")
st.write("You can check the source code on [GitHub](https://github.com/LuisAnibalVasquez/Site-Energy-Intensity-Prediction)")

def main():
    with st.form('prediction_form'):
        
        st.subheader("Enter the input for following features:")

        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                Year_Factor = st.number_input('Year Factor', min_value = 1, max_value = 10, value  = 1, step =1, format = "%i")
            with col2:
                floor_area = st.number_input('floor area', min_value = 1, max_value = 100000000, value  = 1, step =1, format = "%i")
            with col3:
                year_built = st.number_input('year built', min_value = 1600, max_value = 3000, value  = 1600, step =11, format = "%i")
            with col4:
                energy_star_rating = st.number_input('energy star rating', min_value = 0, max_value = 100, value  = 0, step =1, format = "%i")
            with col5:
                ELEVATION = st.number_input('ELEVATION', min_value = 0, max_value = 50000, value  = 0, step =1, format = "%i")                  
            with col6:
                january_min_temp = st.number_input('january min temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
        
        with st.container():
            col7, col8, col9, col10, col11, col12 = st.columns(6)
            with col7:
                january_avg_temp = st.number_input('january avg temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col8:
                january_max_temp = st.number_input('january max temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col9:
                february_min_temp = st.number_input('february min temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col10:
                february_avg_temp = st.number_input('february avg temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col11:                 
                february_max_temp = st.number_input('february max temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col12:
                march_min_temp = st.number_input('march min temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
        
        with st.container():
            col13, col14, col15, col16, col17, col18 = st.columns(6)
            with col13:
                march_avg_temp = st.number_input('march avg temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col14:
                march_max_temp = st.number_input('march max temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col15:
                april_min_temp = st.number_input('april min temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col16:
                april_avg_temp = st.number_input('april avg temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col17:
                april_max_temp = st.number_input('april max temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col18:                                                                                     
                may_min_temp = st.number_input('may min temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")


        with st.container():
            col19, col20, col21, col22, col23, col24 = st.columns(6)
            with col19:
                may_avg_temp = st.number_input('may avg temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col20:
                may_max_temp = st.number_input('may max temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col21:
                june_min_temp = st.number_input('june min temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col22:
                june_avg_temp = st.number_input('june avg temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col23:
                june_max_temp = st.number_input('june max temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col24: 
                july_min_temp = st.number_input('july min temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")

        with st.container():
            col25,col26,col27,col28,col29,col30 = st.columns(6)
            with col25:
                july_avg_temp = st.number_input('july avg temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col26:
                july_max_temp = st.number_input('july max temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col27:
                august_min_temp = st.number_input('august min temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col28:
                august_avg_temp = st.number_input('august avg temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col29:
                august_max_temp = st.number_input('august max temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col30: 
                september_min_temp = st.number_input('september min temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")

        with st.container():
            col31, col32, col33, col34, col35, col36 = st.columns(6)
            with col31:
                september_avg_temp = st.number_input('september avg temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col32:
                september_max_temp = st.number_input('september max temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col33:
                october_min_temp = st.number_input('october min temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col34:
                october_avg_temp = st.number_input('october avg temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col35:
                october_max_temp = st.number_input('october max temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col36: 
                november_min_temp = st.number_input('november min temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")

        with st.container():
            col37, col38, col39, col40,col41,col42 = st.columns(6)
            with col37:
                november_avg_temp = st.number_input('november avg temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col38:
                november_max_temp = st.number_input('november max temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col39:
                december_min_temp = st.number_input('december min temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col40:
                december_avg_temp = st.number_input('december avg temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col41:
                december_max_temp = st.number_input('december max temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")

        with st.container():
            col43,col44,col45,col46,col47, col48 = st.columns(6)
            with col43:
                cooling_degree_days = st.number_input('cooling degree days', min_value = 0, max_value = 100000, value  = 0, step =1, format = "%i")
            with col44:
                heating_degree_days = st.number_input('heating degree days', min_value = 0, max_value = 100000, value  = 0, step =1, format = "%i")
            with col45:
                precipitation_inches = st.number_input('precipitation inches', min_value = 0.0, max_value = 1000.0, value  = 0.0, step =0.01, format = "%f")
            with col46:
                snowfall_inches = st.number_input('snowfall inches', min_value = 0.0, max_value = 1000.0, value  = 0.0, step =0.01, format = "%f")
            with col47:
                snowdepth_inches = st.number_input('snowdepth inches', min_value = 0.0, max_value = 1000.0, value  = 0.0, step =0.01, format = "%f")
            with col48: 
                avg_temp = st.number_input('avg temp', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")


        with st.container():
            col49, col50, col51, col52, col53, col54 = st.columns(6)
            with col49:
                days_below_30F = st.number_input('days below 30F', min_value = 0, max_value = 1000, value  = 0, step =1, format = "%i")
            with col50:
                days_below_20F = st.number_input('days below 20F', min_value = 0, max_value = 1000, value  = 0, step =1, format = "%i")
            with col51:
                days_below_10F = st.number_input('days below 10F', min_value = 0, max_value = 1000, value  = 0, step =1, format = "%i")
            with col52:
                days_below_0F = st.number_input('days below 0F', min_value = 0, max_value = 1000, value  = 0, step =1, format = "%i")
            with col53:
                days_above_80F = st.number_input('days above 80F', min_value = 0, max_value = 1000, value  = 0, step =1, format = "%i")
            with col54: 
                days_above_90F = st.number_input('days above 90F', min_value = 0, max_value = 1000, value  = 0, step =1, format = "%i")

        with st.container():
            col55, col56, col57, col58, col59, col60 = st.columns(6)
            with col55:
                days_above_100F = st.number_input('days above 100F', min_value = 0, max_value = 1000, value  = 0, step =1, format = "%i")
            with col56:
                days_above_110F = st.number_input('days above 110F', min_value = 0, max_value = 1000, value  = 0, step =1, format = "%i")
            with col57:
                direction_max_wind_speed = st.number_input('direction max wind speed', min_value = 0, max_value = 10000, value  = 0, step =1, format = "%i")
            with col58:
                direction_peak_wind_speed = st.number_input('direction peak wind speed', min_value = 0, max_value = 10000, value  = 0, step =1, format = "%i")
            with col59:
                max_wind_speed = st.number_input('max wind speed', min_value = -10.0, max_value = 100.0, value  = 0.0, step =0.01, format = "%f")
            with col60: 
                days_with_fog = st.number_input('days with fog', min_value = 0, max_value = 1000, value  = 0, step =1, format = "%i")

        with st.container():
            col61, col62, col63 = st.columns(3)
            with col61:
                building_class = st.selectbox("Select building class: ", options=Options_building_class)
            with col62:
                State_Factor = st.selectbox("Select state factor: ", options=Options_State_Factor)
            with col63:
                facility_type = st.selectbox("Select facility type: ", options=Options_facility_type)

        submit = st.form_submit_button("Predict")    
        
        if submit:   

            building_class_Commercial, building_class_Residential = one_hot_encoder_building_class(building_class)
            
            label_State_Factor = label_encoder_State_Factor(State_Factor)
            
            target_facility_type = target_encoder_facility_type(facility_type)

            data = np.array([Year_Factor,
                                floor_area,
                                year_built,
                                energy_star_rating,
                                ELEVATION,
                                january_min_temp,
                                january_avg_temp,
                                january_max_temp,
                                february_min_temp,
                                february_avg_temp,
                                february_max_temp,
                                march_min_temp,
                                march_avg_temp,
                                march_max_temp,
                                april_min_temp,
                                april_avg_temp,
                                april_max_temp,
                                may_min_temp,
                                may_avg_temp,
                                may_max_temp,
                                june_min_temp,
                                june_avg_temp,
                                june_max_temp,
                                july_min_temp,
                                july_avg_temp,
                                july_max_temp,
                                august_min_temp,
                                august_avg_temp,
                                august_max_temp,
                                september_min_temp,
                                september_avg_temp,
                                september_max_temp,
                                october_min_temp,
                                october_avg_temp,
                                october_max_temp,
                                november_min_temp,
                                november_avg_temp,
                                november_max_temp,
                                december_min_temp,
                                december_avg_temp,
                                december_max_temp,
                                cooling_degree_days,
                                heating_degree_days,
                                precipitation_inches,
                                snowfall_inches,
                                snowdepth_inches,
                                avg_temp,
                                days_below_30F,
                                days_below_20F,
                                days_below_10F,
                                days_below_0F,
                                days_above_80F,
                                days_above_90F,
                                days_above_100F,
                                days_above_110F,
                                direction_max_wind_speed,
                                direction_peak_wind_speed,
                                max_wind_speed,
                                days_with_fog,
                                building_class_Commercial, 
                                building_class_Residential,
                                label_State_Factor,
                                target_facility_type]).reshape(1,-1)
            
            pred = get_prediction(data=data, model=model)
            
            st.subheader(f"The predicted Site EUI is:  {pred[0]}")

if __name__ == '__main__':
    main()