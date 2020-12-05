import json
import numpy as np
import argparse
import os

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument(
    '-f',
    '--folder',
    default=None,
    type=str
)
argparser.add_argument(
    '-e',
    '--exp',
    default=None,
    type=str
)
argparser.add_argument(
    '-de',
    '--drive-envs',
    nargs='+',
    default=[]
)
args = argparser.parse_args()

success_rates_all = []
success_notrafflight_rates_all = []
success_noviolation_rates_all = []

val_stale_file_name = os.path.join(os.getcwd(), '_logs', args.folder, args.exp, 'validation_Town01_val_stale.csv')
with open(val_stale_file_name, 'rU') as f:
    checkpoint_num = f.readline().rstrip()

for env in args.drive_envs:
    exp_folder = args.folder + '_' + args.exp + '_' + checkpoint_num + '_drive_control_output_' + env
    exp_folder = os.path.join(os.getcwd(), '_benchmarks_results', exp_folder)
    file_name = os.path.join(exp_folder, 'metrics.json')
    summary_file_name = os.path.join(exp_folder, 'summary.csv')

    with open(file_name, 'r') as f:
        file_json = json.load(f)
        
    with open(summary_file_name, 'rU') as f:
        header = f.readline()
        header = header.split(',')
        header[-1] = header[-1][:-1]
        
    result_matrix = np.loadtxt(summary_file_name, delimiter=",", skiprows=1)
        
    weather_list = sorted(list(file_json['episodes_fully_completed'].keys()))
    
    # Compute NoCrash success rates
    success_rates = [0, 0, 0]
    for weather in weather_list:
        for i in range(len(success_rates)):
            success_rates[i] += (sum(file_json['episodes_fully_completed'][weather][i]) / (25.0 * len(weather_list)))
    
    print("NoCrash success rates")
    print(success_rates)
    success_rates_all.append(success_rates)
    
    # Compute NoTrafficLight success rates (NoCrash + cannot cross red light)
    success_notrafflight_rates = [0, 0, 0]
    for weather in weather_list:
        for i in range(len(success_rates)):
            experiment_results_matrix = result_matrix[
                        np.logical_and(result_matrix[:, header.index(
                            'exp_id')] == i, result_matrix[:, header.index('weather')] == int(float(weather)))]
            notrafflight = (experiment_results_matrix[:, header.index('number_red_lights')] == 0)
            # print(np.sum(notrafflight.astype(float)))
            success_notrafflight = np.logical_and(np.array(file_json['episodes_fully_completed'][weather][i]) > 0, notrafflight)
            # print(success_notrafflight)
            success_notrafflight_rates[i] += (np.sum(success_notrafflight.astype(float)) / (25.0 * len(weather_list)))
    
    print("NoTraffLight success rates")
    print(success_notrafflight_rates)
    success_notrafflight_rates_all.append(success_notrafflight_rates)
    
    # Compute Traffic-school success rates (NoCrash + cannot cross red light + no out-of-road infraction)
    success_noviolation_rates = [0, 0, 0]
    for weather in weather_list:
        for i in range(len(success_rates)):
            experiment_results_matrix = result_matrix[
                        np.logical_and(result_matrix[:, header.index(
                            'exp_id')] == i, result_matrix[:, header.index('weather')] == int(float(weather)))]
            notrafflight = (experiment_results_matrix[:, header.index('number_red_lights')] == 0)
            no_violations = np.logical_and(notrafflight, np.array(file_json['intersection_otherlane'][weather][i]) == 0)
            no_violations = np.logical_and(no_violations, np.array(file_json['intersection_offroad'][weather][i]) == 0)
            # print(np.sum(no_violations.astype(float)))
            success_noviolations = np.logical_and(np.array(file_json['episodes_fully_completed'][weather][i]) > 0, no_violations)
            # print(success_noviolations)
            success_noviolation_rates[i] += (np.sum(success_noviolations.astype(float)) / (25.0 * len(weather_list)))
    
    print("Traffic-school success rates")
    print(success_noviolation_rates)
    success_noviolation_rates_all.append(success_noviolation_rates)
    
success_rates_all = np.array(success_rates_all)
success_notrafflight_rates_all = np.array(success_notrafflight_rates_all)
success_noviolation_rates_all = np.array(success_noviolation_rates_all)

# Compute mean +/- s.d. of multiple runs for each evaluation protocol
print("NoCrash mean +/- s.d")
print([100 * np.mean(success_rates_all, axis=0), 100 * np.std(success_rates_all, axis=0, ddof=1)])

print("NoTraffLight mean +/- s.d")
print([100 * np.mean(success_notrafflight_rates_all, axis=0), 100 * np.std(success_notrafflight_rates_all, axis=0, ddof=1)])

print("Traffic-school mean +/- s.d")
print([100 * np.mean(success_noviolation_rates_all, axis=0), 100 * np.std(success_noviolation_rates_all, axis=0, ddof=1)])
    
