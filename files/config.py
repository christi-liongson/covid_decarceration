from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


data_folder = "../data/"
database_name = data_folder + "nc_doc.db"

TABLES = ["APPT7AA1", "APPT9BJ1", "INMT4AA1", "INMT4BB1", "INMT4CA1",
          "INMT9CF1", "OFNT1BA1", "OFNT3AA1", "OFNT3BB1", "OFNT3CE1",
          "OFNT3DE1", "OFNT9BE1"]

holdOut = 0.2
randomState = 1234

#categorical_vars_to_impute = ['RACE','GENDER','ETHNICITY', 'CONTROL_STATUS','OFFENSE_CLEAN', 'Current_Offense_Risk_Level', "Current_Offense_Risk_Level_Lenient","Current_Offense_Risk_Level_Harsh"]

#categorical_vars_to_impute = ['RACE','GENDER','ETHNICITY', 'CONTROL_STATUS','OFFENSE_CLEAN', 'Current_Offense_Risk_Level','year_month']
categorical_vars_to_impute = ['CONTROL_STATUS','OFFENSE_CLEAN', 'Current_Offense_Risk_Level','year_month']


continuous_vars_to_impute = ['INFRACTION_PER_SENT','misd_count','felon_count','sentence_count','age_at_sentence','unemp_rate']
#continuous_vars_to_impute = ['INFRACTION_PER_SENT','misd_count','felon_count','sentence_count']


categorical_vars_one_hot = categorical_vars_to_impute + ['age_cat']
#categorical_vars_one_hot = categorical_vars_to_impute
continuous_vars_normalize = continuous_vars_to_impute 
#continuous_vars_normalize = continuous_vars_to_impute 
#age_unemp_features = ['age_at_sentence', 'age_cat', 'juv_first_offense','unemp_rate', 'year_month']

grouping_target = "binary"
target_vars = ['Recidivate_Risk_Level']
ID_vars = ['ID','COMMITMENT_PREFIX','BIRTH_DATE','EARLIEST_SENTENCE_EFFECTIVE_DT','INMATE_ADMIN_STATUS_CODE','NextPrefix']
keep_vars = target_vars + categorical_vars_to_impute + continuous_vars_to_impute + ID_vars

MODELS = {
    'LogisticRegression': LogisticRegression(random_state=randomState, solver='lbfgs'), 
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=randomState), 
    'RandomForestClassifier': RandomForestClassifier(random_state=randomState),
    'GaussianNB': GaussianNB(),
    'LinearSVC': LinearSVC(random_state=randomState)
}

PARAMS = {
    'LogisticRegression': {
        'penalty': ['l2','none'], 
        'C': [0.01, 0.1, 1, 10, 100], 
        'max_iter':[1000]
        },
    'DecisionTreeClassifier': {
        'criterion': ('gini','entropy'),
        'max_depth': (10,20,30,None),
        'min_samples_split': (5,50,100) ,

    },
    'RandomForestClassifier':{
    'criterion': ('gini','entropy'),
    'n_estimators': (10,20, 30, 100,1000),
    'max_depth': (10,20,30,None),
    'min_samples_split':(5,50,100)
    },
    'GaussianNB': {'priors': [None]},
    'LinearSVC': {'C': [0.01, 0.1, 1, 10, 100]}


}
