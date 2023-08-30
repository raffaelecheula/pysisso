# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pysisso.sklearn import SISSORegressor

# -----------------------------------------------------------------------------
# CONTROL
# -----------------------------------------------------------------------------

# SISSO parameters.
ml_model = 'pysisso'
multitask = True

# Dataset from csv.
csv_file = 'train.csv'
names_key = 'structure_name'
y_dep_key = 'energy_ts'
groups_key = 'reaction'
tasks_keys = ['reaction']

# Regression.
regression = True

# Cross validation.
cross_validation = True
cross_valid_name  = 'crossvalid' # crossvalid | extrapolation
n_splits = 5
print_resul_train = False
print_resul_inter = True
parity_plots = True
violin_plots = True

# Random state.
random_state = 0

# SISSO parameters.
SISSO_exe = "/home/rcheula/Programs/SISSO-3.0.2/bin/SISSO"
opset = '(+)(-)(*)(/)(exp)(exp-)(^-1)(^2)(^3)(sqrt)(cbrt)(log)(|-|)(sin)(cos)'
rung_SISSO = 2
dim_SISSO = 3

# Plot parameters.
fontsize = 16
labelsize = 14

# -----------------------------------------------------------------------------
# READ CSV
# -----------------------------------------------------------------------------

def print_title(title, n_dash=100):
    print('\n'+'-'*n_dash+'\n'+title+'\n'+'-'*n_dash)

print_title('Load data.')

df = pd.read_csv(csv_file)

print(f'Read file: {csv_file}')

# Sort df for multi-task SISSO and extract tasks_array.
df = df.sort_values(by=tasks_keys, ignore_index=True)
tasks_array = df[tasks_keys[0]].to_numpy()
for key in tasks_keys[1:]:
    tasks_array += ' '+df[key].to_numpy()

names = df.pop(names_key).to_numpy()
g_groups = df.pop(groups_key).to_numpy()
y_dep = df.pop(y_dep_key).to_numpy()
X_indep = df.to_numpy()

feature_names = df.columns.to_list()

print(f'N. entries:  {X_indep.shape[0]:4d}')
print(f'N. features: {X_indep.shape[1]:4d}')

# -----------------------------------------------------------------------------
# GET REGRESSOR
# -----------------------------------------------------------------------------

def get_regressor(ml_model, i_split=-1):

    if ml_model == 'pysisso':
        regr = SISSORegressor(
            rung = rung_SISSO,
            opset = opset,
            desc_dim = dim_SISSO,
            task_weighting = 2,
            run_dir = f'SISSO_dir_{i_split+1}',
            clean_run_dir = False,
            SISSO_exe = SISSO_exe,
        )
    
    return regr

# -----------------------------------------------------------------------------
# REGRESSION
# -----------------------------------------------------------------------------

if regression is True:

    print_title('Regression.')

    print(f'\nML model: {ml_model}')

    regr = get_regressor(ml_model=ml_model)

    if multitask is False:
        tasks_array = None

    regr.fit(
        X = X_indep,
        y = y_dep,
        columns = feature_names,
        tasks_array = tasks_array,
    )
    y_pred = regr.predict(
        X = X_indep,
        tasks_array = tasks_array,
    ).reshape(y_dep.shape)

    err_abs = np.abs(y_dep-y_pred)
    err_sqr = np.power((y_dep-y_pred), 2)
    mae = np.average(err_abs)
    rmse = np.sqrt(np.average(err_sqr))

    print('Train')
    print(f'MAE train  = {mae:7.4f} eV')
    print(f'RMSE train = {rmse:7.4f} eV')

# -----------------------------------------------------------------------------
# CROSS VALIDATION
# -----------------------------------------------------------------------------

if cross_validation is True:

    print_title(f'Cross validation ({cross_valid_name}).')

    df_err_data = {
        'Abs Err [eV]': [],
        'ML model': [],
    }

    print(f'\nML model: {ml_model}')

    y_test_tot = []
    y_pred_tot = []
    err_abs_train_tot = []
    err_sqr_train_tot = []
    err_abs_test_tot = []
    err_sqr_test_tot = []

    if cross_valid_name == 'crossvalid':
        cross_valid = StratifiedKFold(
            n_splits = n_splits,
            random_state = random_state,
            shuffle = True,
        )
        train_test_splits = cross_valid.split(X=X_indep, y=g_groups)
    
    elif cross_valid_name == 'extrapolation':
        cross_valid = LeaveOneGroupOut()
        train_test_splits = cross_valid.split(X=X_indep, groups=g_groups)

    i_split = 0
    for train_index, test_index in train_test_splits:

        y_train = y_dep[train_index]
        X_train = X_indep[train_index]
        g_train = g_groups[train_index]
        y_test = y_dep[test_index]
        X_test = X_indep[test_index]
        g_test = g_groups[test_index]

        if multitask is True:
            tasks_train = tasks_array[train_index]
            tasks_test = tasks_array[test_index]
        else:
            tasks_train = None
            tasks_test = None

        regr = get_regressor(ml_model=ml_model, i_split=i_split)

        regr.fit(
            X = X_train,
            y = y_train,
            columns = feature_names,
            tasks_array = tasks_train,
        )
        
        y_pred_train = regr.predict(
            X = X_train,
            tasks_array = tasks_train,
        ).reshape(y_train.shape)
        
        y_pred_test = regr.predict(
            X = X_test,
            tasks_array = tasks_test,
        ).reshape(y_test.shape)

        err_abs_train = np.abs(y_train-y_pred_train)
        err_sqr_train = np.power((y_train-y_pred_train), 2)
        mae_train = np.average(err_abs_train)
        rmse_train = np.sqrt(np.average(err_sqr_train))
        err_abs_train_tot = np.append(err_abs_train_tot, err_abs_train)
        err_sqr_train_tot = np.append(err_sqr_train_tot, err_sqr_train)
            
        err_abs_test = np.abs(y_test-y_pred_test)
        err_sqr_test = np.power((y_test-y_pred_test), 2)
        mae_test = np.average(err_abs_test)
        rmse_test = np.sqrt(np.average(err_sqr_test))
        err_abs_test_tot = np.append(err_abs_test_tot, err_abs_test)
        err_sqr_test_tot = np.append(err_sqr_test_tot, err_sqr_test)

        y_test_tot = np.append(y_test_tot, y_test)
        y_pred_tot = np.append(y_pred_tot, y_pred_test)

        i_split += 1
        if print_resul_inter is True:
            print(f'Split: {i_split}')
            if print_resul_train is True:
                print(f'MAE train  = {mae_train:7.4f} eV')
                print(f'RMSE train = {rmse_train:7.4f} eV')
            print(f'MAE test   = {mae_test:7.4f} eV')
            print(f'RMSE test  = {rmse_test:7.4f} eV')

    print('Average')
    mae_train_tot = np.average(err_abs_train_tot)
    rmse_train_tot = np.sqrt(np.average(err_sqr_train_tot))
    mae_test_tot = np.average(err_abs_test_tot)
    rmse_test_tot = np.sqrt(np.average(err_sqr_test_tot))
    if print_resul_train is True:
        print(f'MAE train  = {mae_train_tot:7.4f} eV')
        print(f'RMSE train = {rmse_train_tot:7.4f} eV')
    print(f'MAE test   = {mae_test_tot:7.4f} eV')
    print(f'RMSE test  = {rmse_test_tot:7.4f} eV')

    df_err_data['Abs Err [eV]'] += list(err_abs_test_tot)
    df_err_data['ML model'] += [ml_model]*len(err_abs_test_tot)

    if parity_plots is True:
        
        plt.style.use('ggplot')
        plt.rc('xtick', labelsize=labelsize) 
        plt.rc('ytick', labelsize=labelsize)
        
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = plt.subplot(111)
        
        plt.title(f"Parity Plot ({ml_model})")
        plt.xlabel("y real [eV]", fontsize=fontsize)
        plt.ylabel("y model [eV]", fontsize=fontsize)
        plt.plot(y_test_tot, y_pred_tot, 'o')
        ax.set_xlim(-2.50, +2.50)
        ax.set_ylim(-2.50, +2.50)
        plt.plot([-2.50, +2.50], [-2.50, +2.50], '--k')
        plt.savefig('parity_plot.png')

    if violin_plots is True:

        df_err = pd.DataFrame(
            data = df_err_data,
            columns = df_err_data.keys(),
        )

        plt.style.use('ggplot')
        plt.rc('xtick', labelsize=labelsize) 
        plt.rc('ytick', labelsize=labelsize)
        fig, ax = plt.subplots()
        sns.set_theme(style="whitegrid")
        sns.violinplot(
            data = df_err,
            y = 'Abs Err [eV]',
            x = 'ML model',
            cut = 3.,
            linewidth = 1,
            palette = 'Set2',
            ax = ax,
        )
        sns.despine()
        ax.set_ylim(0.0, 1.0)
        plt.savefig('violin_plot.png')

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------