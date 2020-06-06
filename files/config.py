
data_folder = "../data/"
database_name = data_folder + "nc_doc.db"

TABLES = ["APPT7AA1", "APPT9BJ1", "INMT4AA1", "INMT4BB1", "INMT4CA1",
          "INMT9CF1", "OFNT1BA1", "OFNT3AA1", "OFNT3BB1", "OFNT3CE1",
          "OFNT3DE1", "OFNT9BE1"]

holdOut = 0.2
randomState = 1234

categorical_vars_to_impute = ['RACE','GENDER','ETHNICITY', 'CONTROL_STATUS','OFFENSE_CLEAN', 'Current_Offense_Risk_Level', "Current_Offense_Risk_Level_Lenient","Current_Offense_Risk_Level_Harsh"]
continuous_vars_to_impute = ['INFRACTION_PER_SENT','misd_count','felon_count','sentence_count','age_at_sentence','unemp_rate']
#continuous_vars_to_impute = ['INFRACTION_PER_SENT','misd_count','felon_count','sentence_count']


categorical_vars_one_hot = categorical_vars_to_impute + ['age_cat','year_month']
#categorical_vars_one_hot = categorical_vars_to_impute
continuous_vars_normalize = continuous_vars_to_impute 
#continuous_vars_normalize = continuous_vars_to_impute 
#age_unemp_features = ['age_at_sentence', 'age_cat', 'juv_first_offense','unemp_rate', 'year_month']
keep_vars = categorical_vars_to_impute + continuous_vars_to_impute + ['ID','COMMITMENT_PREFIX','BIRTH_DATE','EARLIEST_SENTENCE_EFFECTIVE_DT']