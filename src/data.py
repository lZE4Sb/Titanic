import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
    TargetEncoder,
)

cfg = OmegaConf.load('config.yaml')

##### COLUMNS GROUPS #####

FILL_MEDIAN_COLS = ['Fare']

FILL_MOST_COMMON_COLS = ['Pclass', 'SibSp', 'Parch', 'Embarked']

CAT_FEATURES = ['Pclass', 'SibSp', 'Parch', 'Embarked', 'Sex', 'Initial']

##### SKEWED COLS #####

QUANTILE_COLS = ['Fare']

##### DATA PREPARATION FUNCTIONS #####

def load_data(cfg):
    '''Load the dataset from the CSV file.'''
    df = pd.read_csv(cfg.paths.train_csv, index_col=cfg.data.id_col)
    if cfg.general.debug: #takes first 10 percent of dataset
        part = round(len(df) / 10)
        df = df.head(part)
    return df

def stateless_preprocessing(df):
    df['Name'].fillna(' Mr.') # filling potential Name misses with most common initial
    df['Initial']=df['Name'].str.extract('([A-Za-z]+)\.')
    df['Initial'] = df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Other','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'])
    df = df.drop(['Name', 'Cabin', 'Ticket'], axis=1) # idk, how to use Cabin and Ticket
    # Filling Age misses with mean value by initials
    df.loc[(df['Age'].isnull()) & (df['Initial'] == 'Master'), 'Age'] = 5
    df.loc[(df['Age'].isnull()) & (df['Initial'] == 'Miss'), 'Age'] = 22
    df.loc[(df['Age'].isnull()) & (df['Initial'] == 'Mr'), 'Age'] = 33
    df.loc[(df['Age'].isnull()) & (df['Initial'] == 'Mrs'), 'Age'] = 36
    df.loc[(df['Age'].isnull()) & (df['Initial'] == 'Other'), 'Age'] = 46
    # filling potential Sex misses by initial
    df.loc[(df['Sex'].isnull()) & (df['Initial'] == 'Master'), 'Sex'] = 'male'
    df.loc[(df['Sex'].isnull()) & (df['Initial'] == 'Miss'), 'Sex'] = 'female'
    df.loc[(df['Sex'].isnull()) & (df['Initial'] == 'Mr'), 'Sex'] = 'male'
    df.loc[(df['Sex'].isnull()) & (df['Initial'] == 'Mrs'), 'Sex'] = 'female'
    df.loc[(df['Sex'].isnull()) & (df['Initial'] == 'Master'), 'Sex'] = 'male'
    df.loc[(df['Sex'].isnull()) & (df['Initial'] == 'Other'), 'Sex'] = 'male' #in Other we have mostly males
    # coding Sex
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    return df

def make_test_split(cfg, df):
    '''Split the dataset into training and testing sets if test size > 0'''
    if cfg.data.test_size > 0:
        df, test = train_test_split(df, 
                                    test_size=cfg.data.test_size,
                                    random_state=cfg.general.seed)
    else:
        test = None
    return df, test

def make_pipeline(cfg, df):    
    imputer_transformer = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), FILL_MEDIAN_COLS), 
            ('cat', SimpleImputer(strategy='most_frequent'), FILL_MOST_COMMON_COLS) 
        ], verbose_feature_names_out=False, 
            remainder='passthrough').set_output(transform='pandas')

    target_encoder = TargetEncoder(
        target_type = 'binary',
        smooth = 20.0,
        cv = 5,
        shuffle = True,
        random_state = cfg.general.seed,
        )

    target_transformer = ColumnTransformer(
        transformers=[
            ('cat', target_encoder, CAT_FEATURES)],
            verbose_feature_names_out=False, 
            remainder='passthrough').set_output(transform='pandas')

    quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=cfg.general.seed)

    skewness_transformer = ColumnTransformer(
        transformers=[
                ('quant', quantile_transformer, QUANTILE_COLS)],
            verbose_feature_names_out=False, 
            remainder='passthrough').set_output(transform='pandas')

    standard_transformer = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), df.drop(cfg.data.target_col, axis=1).columns) # i'm gonna scale all, so no need to choose
        ],
        verbose_feature_names_out=False,
        remainder='passthrough' 
    ).set_output(transform='pandas')

    categorical_encoder = OrdinalEncoder(
        categories = 'auto',
        handle_unknown='use_encoded_value', # this encoder for XGB only, cos he cant handle unknown cat values
        unknown_value=-1,
        encoded_missing_value=-2
    )

    categorical_transformer = ColumnTransformer(
        transformers=[
            ('cat', categorical_encoder, CAT_FEATURES)],
            verbose_feature_names_out=False, 
            remainder='passthrough').set_output(transform='pandas') 

    if cfg.data.pipeline == 'linear':
        pipeline = Pipeline(steps=[
            ('imputer', imputer_transformer),
            ('target', target_transformer),
            ('skewness', skewness_transformer),
            ('scaler', standard_transformer)
        ])
    elif cfg.data.pipeline == 'boosting':
        if cfg.model_class in ('xgb', 'lgbm'):
            pipeline = Pipeline(steps=[
                ('cat', categorical_transformer)
            ]) # for XGB and LGBM, cos they cant handle unknown cat values
        else:
            pipeline = Pipeline([
                ('skip', 'passthrough')])

    return pipeline

def create_splits(cfg, df, col_name = 'fold'
) -> pd.DataFrame: 
    '''
    Creates train/validation or k-fold split column for the given DataFrame.
    Uses bin stratification for continuous target if specified.
    '''
    df = df.copy()

    target_col = cfg.data.target_col
    split_strategy = cfg.data.split_strategy
    n_splits = cfg.data.n_splits
    val_size = cfg.data.val_size
    stratify = cfg.data.stratify
    n_bins_stratify = cfg.data.n_splits
    random_state = cfg.general.seed
    
    # 1. Stratification
    stratify_target = None
    if stratify:
        if target_col is None or target_col not in df.columns:
            raise ValueError('Need to specify `target_col`')
        
        # Bin continuous target into quantiles for stratification if it's numeric and has many unique values
        if np.issubdtype(df[target_col].dtype, np.number) and df[target_col].nunique() > 20:
            stratify_target = pd.qcut(df[target_col], q=n_bins_stratify, labels=False, duplicates='drop')
        else:
            stratify_target = df[target_col]

    # 2. Train / Validation
    if split_strategy == 'train_val':
        df[col_name] = 'train'
        
        if stratify:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
            _, val_idx = next(splitter.split(df, stratify_target))
        else:
            splitter = ShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
            _, val_idx = next(splitter.split(df))
            
        df.iloc[val_idx, df.columns.get_loc(col_name)] = 'val'

    # 3. K-Fold Cross-Validation
    elif split_strategy == 'kfold':
        df[col_name] = -1
        
        if stratify:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_generator = skf.split(df, stratify_target)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_generator = kf.split(df)
            
        for fold, (_, val_idx) in enumerate(split_generator):
            df.iloc[val_idx, df.columns.get_loc(col_name)] = fold

    else:
        raise ValueError(f"Unknown strategy '{split_strategy}'. Available strategies: 'kfold', 'train_val'")

    return df

### IN-CV CYCLE FUNCTION ###

def prepare_data_by_folds(cfg, df, pipeline, fold=0):
    '''
    Prepares training and validation data through
    preprocessing pipeline for a specific fold.
    
    Args:
        cfg: config dict
        df: DataFrame containing the data with a 'fold' column
        pipeline: Preprocessing sklearn pipeline to transform features
        fold: Current fold number (int)
        '''
    target_col = cfg.data.target_col

    X_train = df[df['fold'] != fold].drop(['fold', target_col], axis=1)
    y_train = df[df['fold'] != fold][target_col]
    X_val = df[df['fold'] == fold].drop(['fold', target_col], axis=1)
    y_val = df[df['fold'] == fold][target_col]

    X_train = pipeline.fit_transform(X_train, y_train)
    X_val = pipeline.transform(X_val)

    return X_train, y_train, X_val, y_val


