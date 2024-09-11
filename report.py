import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
import os
import json
from scipy import stats
import math
from scipy.optimize import fsolve,minimize


global config
global phase
global exps

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
f = open("./experiments.json") 
exps = json.load(f)
exp_id = 6
# phase = "baseline"
# phase = "dynamic-ist"
# phase = "best-init"
phase = "temp"
model = "gsr3"
config = exps[model][exp_id - 1]
logs_path = "./logs"

def model_name():
    return f'Google Speech {config["layers"]} Layer - {config["no_nodes"]} Nodes'

def test_acc_path(exp, phasee, node):
    return f'{logs_path}/gsr{config["layers"]}/{phase}/{config["no_nodes"]}-{config["exp"]}/{node}/DNN_speech_{config["layers"]}_layer_BN_{config["epochs"]}_{config["model_size"]}_cascaded_{config["no_nodes"]}_{config["sync_freq"]}_test_acc.log'
    
def test_loss_path(exp, node):
    return f'{logs_path}/gsr{config["layers"]}/{phase}/{config["no_nodes"]}-{config["exp"]}/{node}/DNN_speech_{config["layers"]}_layer_BN_{config["epochs"]}_{config["model_size"]}_cascaded_{config["no_nodes"]}_{config["sync_freq"]}_test_loss.log'

def train_time_path(exp, phasee, node):
    return f'{logs_path}/gsr{config["layers"]}/{phase}/{config["no_nodes"]}-{config["exp"]}/{node}/DNN_speech_{config["layers"]}_layer_BN_{config["epochs"]}_{config["model_size"]}_cascaded_{config["no_nodes"]}_{config["sync_freq"]}_train_time.log'

def sync_time_path(exp, node):
    return f'{logs_path}/gsr{config["layers"]}/{phase}/{config["no_nodes"]}-{config["exp"]}/{node}/DNN_speech_{config["layers"]}_layer_BN_{config["epochs"]}_{config["model_size"]}_cascaded_{config["no_nodes"]}_{config["sync_freq"]}_sync_time.log'

def coefs_path(exp, node):
    return f'{logs_path}/gsr{config["layers"]}/{phase}/{config["no_nodes"]}-{config["exp"]}/{node}/DNN_speech_{config["layers"]}_layer_BN_{config["epochs"]}_{config["model_size"]}_cascaded_{config["no_nodes"]}_{config["sync_freq"]}_coefs.log'


def accuracy_levels():
    return options[config["model"]]["accs"]

options = {
    "gsr": {
        "accs": [0.63, 0.75, 0.795],
        "test_acc_path": f'{logs_path}/ist-{config["layers"]}layer-{config["no_nodes"]}/DNN_speech_{config["layers"]}_layer_BN_{config["epochs"]}_{config["model_size"]}_cascaded_{config["no_nodes"]}_{config["sync_freq"]}_test_acc.log',
        "train_time_path": f'{logs_path}/ist-{config["layers"]}layer-{config["no_nodes"]}/DNN_speech_{config["layers"]}_layer_BN_{config["epochs"]}_{config["model_size"]}_cascaded_{config["no_nodes"]}_{config["sync_freq"]}_train_time.log',
        "sync_time_path": f'{logs_path}/ist-{config["layers"]}layer-{config["no_nodes"]}/DNN_speech_{config["layers"]}_layer_BN_{config["epochs"]}_{config["model_size"]}_cascaded_{config["no_nodes"]}_{config["sync_freq"]}_train_time.log',
        "model_name": f'Google Speech {config["layers"]} Layer - {config["no_nodes"]} Nodes'
    }
}

gpu_filepath = f'gpu_logs\{config["model"]}\\{config["no_nodes"]}nodes\\'

def read_logs(filename, delimiter):
    reader = csv.reader(open(filename), delimiter=delimiter)
    items = []
    logs = []
    for line in reader:
        items.append(line)
    for item in items[0]:
        if item != "" and item != " ":
            logs.append(float(item))
    return logs


def read_sync_time_logs():
    nodes = ["server"]
    logs = {}
    for i in range(config["no_nodes"] - 1):
        nodes.append(f"worker{i+1}")
    
    for node in nodes:
        sync_logs = read_logs(sync_time_path(config["exp"], node), " ")[0:last_sync_to_keep(phase) + 1]
        logs[node] = sync_logs

    df = pd.DataFrame(logs, index=range(1,last_sync_to_keep(phase) + 2))
    df.index.name = "sync_no"
    return df


def read_coef_logs():
    nodes = ["server"]
    logs = {}
    for i in range(config["no_nodes"] - 1):
        nodes.append(f"worker{i+1}")
    
    for node in nodes:
        coef_logs = read_logs(coefs_path(config["exp"], node), " ")[0:last_sync_to_keep(phase) + 1]
        logs[node] = coef_logs

    df = pd.DataFrame(logs, index=range(1,last_sync_to_keep(phase) + 2))
    df.index.name = "sync_no"
    return df


def last_epoch_to_keep(phasee):
    global phase
    global config
    test_acc_logs = read_logs(test_acc_path(config["exp"], phase, "server"), " ")
    good_accs = [i for i in range(len(test_acc_logs)) if test_acc_logs[i] >= config["acc_levels"][phase][-1]]
    last_epoch = good_accs[0] if len(good_accs) > 0 else len(test_acc_logs) - 1
    return last_epoch


def last_sync_to_keep(phasee):
    global phase
    global config
    iters_per_epoch = (76364/4)//config["batch_size"]
    syncs_per_epoch = math.ceil(iters_per_epoch/config["sync_freq"])
    return last_epoch_to_keep(phase) * syncs_per_epoch


def read_test_acc_logs():
    global phase
    test_acc_logs = read_logs(test_acc_path(config["exp"], phase, "server"), " ")
    train_time_logs = read_logs(train_time_path(config["exp"], phase, "server"), " ")
    # last_epoch = test_acc_logs.find(test_acc_logs>=options[config["model"]]["accs"][-1])[0]
    last_epoch = last_epoch_to_keep(phase)
    test_acc_logs = test_acc_logs[0:last_epoch + 1]
    train_time_logs = train_time_logs[0:last_epoch + 1]
    columns = {"train_time": train_time_logs, "test_acc": test_acc_logs}
    df = pd.DataFrame(columns, index=range(len(test_acc_logs)))
    df.index.name = "epoch_no"
    df['train_time'] = df['train_time'].cumsum()
    # df['train_time'] -= df["train_time"].min()
    return df



def plot_coef_time():

    sync_df = read_sync_time_logs()
    coef_df = read_coef_logs()

    nodes = ['server', 'worker1', 'worker2', 'worker3']

    ai = []
    bi = []
    ci = []
    for node in nodes:
        x = coef_df[node].to_numpy()
        y = sync_df[node].to_numpy()
        # a, b = np.polyfit(x, y, 1)
        a, b, c = np.polyfit(x, y, 2)
        ai.append(a)
        bi.append(b)
        ci.append(c)
        cont_x = np.linspace(0, 1, 1000)
        plt.scatter(x, y, alpha=0.2)
        plt.plot(cont_x, a*cont_x**2 + b*cont_x + c, label=node)
        # plt.plot(cont_x, a*cont_x + b, label=node)

    np.set_printoptions(precision=8)
    a = np.array(ai)
    b = np.array(bi)
    c = np.array(ci)

    def ti(i, p):
        return a[i]*p**2 + b[i]*p + c[i]
    
    def objective(p):
        p1, p2, p3, p4 = p
        t1_val = ti(0, p1)
        t2_val = ti(1, p2)
        t3_val = ti(2, p3)
        t4_val = ti(3, p4)
        return (t1_val - t2_val)**2 + (t1_val - t3_val)**2 + (t1_val - t4_val)**2

    # Equality constraint: p1 + p2 + p3 + p4 = 1
    def constraint_sum(p):
        return np.sum(p) - 1

    # Bounds for each p_i (0 < p_i < 1)
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]

    # Initial guess for p1, p2, p3, p4
    initial_guess = [0.25, 0.25, 0.25, 0.25]

    # Set up the constraints (sum constraint)
    constraints = [{'type': 'eq', 'fun': constraint_sum}]

    # Perform the optimization
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)

    # Print the optimized p1, p2, p3, p4
    if result.success:
        p1_opt, p2_opt, p3_opt, p4_opt = result.x
        print(f'Optimized Coefs: {p1_opt},{p2_opt},{p3_opt},{p4_opt}')
        print(ti(0, p1_opt), ti(1, p2_opt), ti(2, p3_opt), ti(3, p4_opt))
    else:
        print("Optimization failed:", result.message)

    # k = (1 + np.sum(b/a))/np.sum(1/a)
    # c = (k-b)/a
    # print(c)
    plt.legend(loc="upper left")
    plt.title(f"Training Time vs Model Size Portion - Exp{exp_id}")
    plt.xlabel("Model Size Portion")
    plt.ylabel("Training Time")
    plt.xlim(0,1)
    plt.savefig(f"figures/{model}/train_time_v_model_size_portion-exp{exp_id}.png", bbox_inches='tight', pad_inches=0.1, dpi=199)
    plt.show()
    plt.clf()
    
    # fig, ax = plt.subplots(nrows=2, ncols=2)
    # i = 0
    # for row in ax:
    #     for col in row:
    #         x = coef_df[nodes[i]].to_numpy()
    #         y = sync_df[nodes[i]].to_numpy()
    #         slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    #         # col.scatter(x=coef_df[nodes[i]], y=sync_df[nodes[i]])
    #         axes = plt.gca()
    #         x_vals = np.array(axes.get_xlim())
    #         y_vals = intercept + slope * x_vals
    #         col.plot(x_vals, y_vals, '--', color='red')
    #         i += 1
    # plt.show()


def plot_std_of_sync_times(df, p, e):
    std = df.std(axis=1)
    mean_std = std.mean()
    std.plot()
    plt.axhline(y = mean_std, color = 'r', linestyle = '-') 
    plt.xlabel("Sync Number")
    plt.ylabel("Standard Deviation of Training Times")
    plt.title(f"Exp{e} - STD deviation of training times - {p}")
    plt.savefig(f"figures/{model}/{p}/exp{e}/std.png", bbox_inches='tight', pad_inches=0.1, dpi=199)
    plt.clf()
    # plt.show()


def report_plot_idle_times(df, p, e):
    print("\n\nNodes Training Times:")
    print(df.sum())
    nodes = ["server", "worker1", "worker2", "worker3"]
    max_times = df.max(axis=1)
    total_training_time = max_times.sum()
    print(f"Total Training Time: {total_training_time}")
    idle_times = df.apply(lambda x: max_times - x)

    total_idle_time = idle_times.sum(axis=0)
    idle_time_percentage = total_idle_time / total_training_time
    print(f"Nodes Idle Time: {total_idle_time}")
    print(f"Nodes Idle Percentage: {idle_time_percentage}\n\n")

    ax = max_times.plot.line()
    ax.legend(["total training time"])
    
    for node in nodes:
        idle_times[node].plot.area(ax=ax, stacked=False, legend=node, alpha=0.35)
    
    plt.title(f"Exp{e} - Workers' Idle Times - {p}")
    plt.xlabel("Sync Number")
    plt.ylabel("Time(s)")
    plt.savefig(f"figures/{model}/{p}/exp{e}/idle_times.png", bbox_inches='tight', pad_inches=0.1, dpi=199)
    plt.clf()
    # plt.show()


def plot_training_times(df, p, e):
    # df['min_train_time'] = df.min(axis=1)
    # df['max_train_time'] = df.max(axis=1)
    df.plot.area(stacked=False, alpha=0.35)
    # df.plot.line(y=['min_train_time', 'max_train_time'])
    plt.xlabel("Sync Number")
    plt.ylabel("Training Time")
    # plt.legend(['Fastest Worker', 'Slowest Worker'])
    # plt.title("Fastest Worker vs Slowest Worker")
    plt.savefig(f"figures/{model}/{p}/exp{e}/training_times.png", bbox_inches='tight', pad_inches=0.1, dpi=199)
    plt.clf()
    # plt.show()


def plot_fast_v_slow(df, p, e):
    df['min_train_time'] = df.min(axis=1)
    df['max_train_time'] = df.max(axis=1)
    df.plot.line(y=['min_train_time', 'max_train_time'])
    plt.legend(['Fastest Worker', 'Slowest Worker'])
    plt.title(f"Exp{e} - Fastest vs Slowest Worker - {p}")

    plt.xlabel("Sync Number")
    plt.ylabel("Training Time")
    plt.savefig(f"figures/{model}/{p}/exp{e}/fast_v_slow.png", bbox_inches='tight', pad_inches=0.1, dpi=199)
    plt.clf()
    # plt.show()


def report_plot_accuracy(phases):
    global phase
    phase = phases[0]
    df1 = read_test_acc_logs()
    ax = df1.plot.line(x='train_time', y='test_acc', style='.-')
    acc_levels = config["acc_levels"][phase]
    for i in range(0,len(phases)):
        phase = phases[i]
        df = read_test_acc_logs()
        if i > 0:
            df.plot.line(x='train_time', y='test_acc', ax=ax, legend=phases[i], style='.-')
        time_to_accs = [[df['train_time'].iloc[i] for i in range(len(df['test_acc'])) if df["test_acc"].iloc[i] >= acc_levels[j]][0] for j in range(len(acc_levels))]
        print(f"For {phases[i]}:\n{time_to_accs}")
    
    for acc_level in config["acc_levels"][phases[-1]]:
        plt.axhline(y = acc_level, color = 'gray', linestyle = '--')

    ax.legend(phases)
    plt.xlabel("Time (s)")
    plt.ylabel("Test Accuracy")
    plt.title(f"Exp{exp_id} - Accuracy vs Training Time")
    plt.savefig(f"figures/{model}/acc_v_time/exp{exp_id}-acc_v_time.png", bbox_inches='tight', pad_inches=0.1, dpi=199)
    # plt.show()
    plt.clf()


# sync_df = read_sync_time_logs()

report_plot_accuracy(['baseline', 'dynamic-ist', 'best-init'])

# for p in ['baseline', 'dynamic-ist', 'best-init']:
#     phase = p
#     sync_df = read_sync_time_logs()
#     plot_std_of_sync_times(sync_df, p, exp_id)
#     report_plot_idle_times(sync_df, p, exp_id)
#     plot_training_times(sync_df, p, exp_id)
#     plot_fast_v_slow(sync_df, p, exp_id)


# plot_std_of_sync_times(sync_df, phase, exp_id)
# report_plot_idle_times(sync_df, phase, exp_id)
# plot_training_times(sync_df, phase, exp_id)
# plot_fast_v_slow(sync_df, phase, exp_id)


# plot_coef_time()
