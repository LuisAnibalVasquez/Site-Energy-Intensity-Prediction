import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def ordinal_encoder(input_val, feats): 
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value

def one_hot_encoder_building_class(val):
    if val == 'Commercial':
        res = 1
        res2 = 0
    else:
        res = 0
        res2 = 1 
    return res, res2

def label_encoder_State_Factor(val):
    if val =='State_1':
        res = 0    
    elif val  == 'State_2':
        res = 1        
    elif val == 'State_4':
        res = 2        
    elif val == 'State_6':
        res = 3        
    elif val == 'State_8':
        res = 4        
    elif val  == 'State_10':
        res = 5        
    else:
        res = 6

    return res


def target_encoder_facility_type(val):
    if val =='Grocery_store_or_food_market':
        res = 241.13516215
    elif val =='Warehouse_Distribution_or_Shipping_center':
        res = 39.559542
    elif val =='Retail_Enclosed_mall' :
        res = 100.96510294
    elif val =='Education_Other_classroom':
        res = 69.44153145      
    elif val =='Warehouse_Nonrefrigerated' :
        res = 38.20939936     
    elif val =='Warehouse_Selfstorage':
        res = 21.58928351      
    elif val =='Office_Uncategorized' :
        res = 77.0743892       
    elif val =='Data_Center' :
        res = 254.4093011       
    elif val =='Commercial_Other':
        res = 92.64203745        
    elif val =='Mixed_Use_Predominantly_Commercial':
        res = 69.13340046       
    elif val =='Office_Medical_non_diagnostic' :
       res = 116.76229367    
    elif val =='Education_College_or_university':
       res = 108.62911621     
    elif val =='Industrial' :
        res = 125.34529468
    elif val =='Laboratory':
        res = 329.40735239
    elif val =='Public_Assembly_Entertainment_culture':
        res = 118.89593034
    elif val =='Retail_Vehicle_dealership_showroom':
        res = 46.80256442
    elif val =='Retail_Uncategorized':
        res = 80.90285378
    elif val =='Lodging_Hotel':
        res = 104.93499689
    elif val =='Retail_Strip_shopping_mall':
        res = 46.1727995
    elif val =='Education_Uncategorized' :
        res = 110.41593618
    elif val =='Health_Care_Inpatient':
        res = 248.34059707
    elif val =='Public_Assembly_Drama_theater' :
        res = 80.92788627
    elif val =='Public_Assembly_Social_meeting':
        res = 78.92602891
    elif val =='Religious_worship' :
        res = 44.56067529
    elif val =='Mixed_Use_Commercial_and_Residential':
        res = 89.52979835
    elif val =='Office_Bank_or_other_financial':
        res = 89.89573416
    elif val =='Parking_Garage':
        res = 67.35124084
    elif val =='Commercial_Unknown' :
        res = 113.13473083
    elif val =='Service_Vehicle_service_repair_shop':
        res = 137.59533585
    elif val =='Service_Drycleaning_or_Laundry':
        res = 72.47718189
    elif val =='Public_Assembly_Recreation':
        res = 114.9988931
    elif val =='Service_Uncategorized':
        res = 113.31254726
    elif val =='Warehouse_Refrigerated':
        res = 96.52402631
    elif val =='Food_Service_Uncategorized' :
        res = 103.38992634
    elif val =='Health_Care_Uncategorized':
        res = 179.57582772
    elif val =='Food_Service_Other' :
        res = 59.88201925
    elif val =='Public_Assembly_Movie_Theater':
        res = 100.42667466
    elif val =='Food_Service_Restaurant_or_cafeteria':
        res = 194.22329978
    elif val =='Food_Sales':
        res = 136.43041262
    elif val =='Public_Assembly_Uncategorized' :
        res = 70.31472411
    elif val =='Nursing_Home':
        res = 131.31388678
    elif val =='Health_Care_Outpatient_Clinic':
        res = 103.05383808
    elif val =='Education_Preschool_or_daycare':
        res = 60.97747158
    elif val =='5plus_Unit_Building' :
        res = 36.73778472
    elif val =='Multifamily_Uncategorized':
        res = 83.87852152
    elif val =='Lodging_Dormitory_or_fraternity_sorority':
        res = 81.59582508
    elif val =='Public_Assembly_Library' :
        res = 105.84908929
    elif val =='Public_Safety_Uncategorized':
        res = 83.60669582
    elif val =='Public_Safety_Fire_or_police_station':
        res = 131.1227119
    elif val =='Office_Mixed_use':
        res = 82.37114456
    elif val =='Public_Assembly_Other':
        res = 126.70407326
    elif val =='Public_Safety_Penitentiary':
        res = 157.26248866
    elif val =='Health_Care_Outpatient_Uncategorized':
        res = 171.8666323
    elif val =='Lodging_Other':
        res = 120.8021388
    elif val =='Mixed_Use_Predominantly_Residential' :
        res = 82.38819214
    elif val =='Public_Safety_Courthouse':
        res = 100.26038947
    elif val =='Public_Assembly_Stadium':
        res = 101.17956848
    elif val =='Lodging_Uncategorized':
        res = 79.66784441
    elif val =='2to4_Unit_Building':
        res = 31.87691514
    else:
        res = 35.93643511
    return res

def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)