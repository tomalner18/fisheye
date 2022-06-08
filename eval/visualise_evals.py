from select import select
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the visualiser config file
CONFIG_PATH = "eval_config.yaml"


# Load parameters

def load_yaml(config_path):
    with open(config_path, 'r') as f:
        config = list(yaml.safe_load_all(f))[0]
    return config


def get_eval_dict(config_yaml):
    # Check if "all" is set for experiments, and load all experiment folder in that case
    if("all" in config_yaml["experiments"]):
        experiment_folders = [f for f in os.listdir(
            "./") if os.path.isdir(os.path.join("./", f))]
        config_yaml["experiments"] = experiment_folders
        print(experiment_folders)

    # This is the dict that is going to contain all the extracted eval data from all experiments
    eval_dict = {}

    # Iterate through all experiments
    for experiment_name in config_yaml["experiments"]:

        # Opening the metrics file of this experiment
        metrics_filepath = os.path.join(experiment_name, "metrics.yaml")
        if(os.path.isfile(metrics_filepath)):

            experiment_metrics_yaml = load_yaml(metrics_filepath)

            eval_dict[experiment_name] = {}

            # Iterate over the evaluation results for all the datasets
            for dataset_eval_name, dataset_eval_metrics in experiment_metrics_yaml["PerDs"].items():
                eval_dict[experiment_name][dataset_eval_name] = []

                # Filter out only the relevant metrics here
                for selected_metric in config_yaml["metrics"]:
                    filtered_metric_results = dataset_eval_metrics[selected_metric[0]
                                                                   ][selected_metric[1]]
                    eval_dict[experiment_name][dataset_eval_name] = {}

                    # Go through all the classes
                    for class_name, class_eval_result in filtered_metric_results.items():
                        # Construct the eval metric dict
                        eval_dict[experiment_name][dataset_eval_name][class_name] = class_eval_result

    return(eval_dict)


def create_bar_chart_all_datasets_all_classes(eval_dict):
    labels = eval_dict.keys()
    x = np.arange(len(labels))  # the label locations
    all_averages = []

    # Create a bar plot for each class
    for experiments, results in eval_dict.items():
        all_averages.append(results["ALL"]["all"])

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, all_averages, 0.25, label='mAP all classes')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('mAP')
    ax.set_title('mAP average for all classes and all datasets')
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)

    fig.tight_layout()
    plt.ylim([min(all_averages)*0.95, max(all_averages)*1.05])
    plt.xticks(rotation=45, ha="right")
    plt.subplots_adjust(bottom=0.5)
    plt.savefig("./output_figures/all_datasets_all_classes.png")

    print("Saved Figure (mAP average for all classes and all datasets) successfully!")


def create_group_3_bar_chart_all_datasets(
        eval_dict,
        class_names=["person", "bicycle and person", "motorcycle and person"],
        chart_title="mAP average for VRUs for all datasets",
        save_file="./output_figures/all_datasets_VRUs.png"):

    labels = eval_dict.keys()
    x = np.arange(len(labels))  # the label locations

    class_1_name = class_names[0]
    class_1 = []
    class_2_name = class_names[1]
    class_2 = []
    class_3_name = class_names[2]
    class_3 = []

    # Create a bar plot for each class
    for experiments, results in eval_dict.items():
        class_1.append(results["ALL"][class_1_name])
        class_2.append(results["ALL"][class_2_name])
        class_3.append(results["ALL"][class_3_name])

    fig, ax = plt.subplots()

    ax.set_axisbelow(True)
    plt.grid()

    rects_1 = ax.bar(x + 0.00, class_1, color='b',
                     width=0.25, label=class_1_name)
    rects_2 = ax.bar(x + 0.25, class_2, color='g',
                     width=0.25, label=class_2_name)
    rects_3 = ax.bar(x + 0.50, class_3, color='r',
                     width=0.25, label=class_3_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('mAP')
    ax.set_title(chart_title)
    ax.set_xticks(x, labels)
    ax.legend()

    fig.tight_layout()
    plt.ylim([min(class_1+class_2+class_3)*0.95,
             max(class_1+class_2+class_3)*1.05])
    plt.xticks(rotation=45, ha="right")
    plt.subplots_adjust(bottom=0.5)
    plt.savefig(save_file)

    print("Saved Figure ", chart_title, " successfully!")


def create_group_3_bar_chart_3_datasets(
        eval_dict,
        class_names=("person", "bicycle and person", "motorcycle and person"),
        dataset_names=("bdd_day", "bdd_night", "bdd_dawn", "woodscape"),
        chart_title="mAP average for VRUs for all datasets",
        save_file="./output_figures/bdd_datasets_VRUs.png"):

    labels = eval_dict.keys()
    x = np.arange(len(labels))  # the label locations

    class_1_name = class_names[0]
    class_1_dataset_1 = []
    class_1_dataset_2 = []
    class_1_dataset_3 = []
    class_2_name = class_names[1]
    class_2_dataset_1 = []
    class_2_dataset_2 = []
    class_2_dataset_3 = []
    class_3_name = class_names[2]
    class_3_dataset_1 = []
    class_3_dataset_2 = []
    class_3_dataset_3 = []

    # Create a bar plot for each class
    for experiments, results in eval_dict.items():
        class_1_dataset_1.append(results[dataset_names[0]][class_1_name])
        class_1_dataset_2.append(results[dataset_names[1]][class_1_name])
        class_1_dataset_3.append(results[dataset_names[3]][class_1_name])
        class_2_dataset_1.append(results[dataset_names[0]][class_2_name])
        class_2_dataset_2.append(results[dataset_names[1]][class_2_name])
        class_1_dataset_3.append(results[dataset_names[3]][class_2_name])
        class_3_dataset_1.append(results[dataset_names[0]][class_3_name])
        class_3_dataset_2.append(results[dataset_names[1]][class_3_name])
        class_3_dataset_2.append(results[dataset_names[3]][class_3_name])

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)

    ax.set_axisbelow(True)
    plt.grid()

    rects_1_1 = ax.bar(x - 0.05, class_1_dataset_1, color='lightblue',
                       width=0.1, label=class_1_name+" ("+dataset_names[0]+")")
    rects_1_2 = ax.bar(x + 0.05, class_1_dataset_2, color='darkblue',
                       width=0.1, label=class_1_name+" ("+dataset_names[1]+")")
    rects_1_3 = ax.bar(x + 0.05, class_1_dataset_3, color='darkblue',
                       width=0.1, label=class_1_name+" ("+dataset_names[3]+")")

    rects_1_1 = ax.bar(x + 0.2, class_2_dataset_1, color='lightgreen',
                       width=0.1, label=class_2_name+" ("+dataset_names[0]+")")
    rects_1_2 = ax.bar(x + 0.3, class_2_dataset_2, color='darkgreen',
                       width=0.1, label=class_2_name+" ("+dataset_names[1]+")")
    rects_1_3 = ax.bar(x + 0.05, class_2_dataset_3, color='darkblue',
                       width=0.1, label=class_2_name+" ("+dataset_names[3]+")")

    rects_1_1 = ax.bar(x - 0.3, class_2_dataset_1, color='salmon',
                       width=0.1, label=class_3_name+" ("+dataset_names[0]+")")
    rects_1_2 = ax.bar(x - 0.2, class_2_dataset_2, color='darkred',
                       width=0.1, label=class_3_name+" ("+dataset_names[1]+")")
    rects_1_3 = ax.bar(x + 0.05, class_3_dataset_3, color='darkblue',
                       width=0.1, label=class_3_name+" ("+dataset_names[3]+")")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('mAP')
    ax.set_title(chart_title)
    ax.set_xticks(x, labels)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
              fancybox=True, shadow=True, ncol=3)

    fig.tight_layout()
    all_values = class_1_dataset_1+class_1_dataset_2+class_1_dataset_3+class_2_dataset_1 + \
        class_2_dataset_2+class_2_dataset_3+class_3_dataset_1+class_3_dataset_2+ class_3_dataset_3
    plt.ylim([min(all_values)*0.95, max(all_values)*1.05])
    plt.xticks(rotation=45, ha="right")
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(save_file)

    print("Saved Figure ", chart_title, " successfully!")


# Maybe use some sort of decorators to adapt this function for different graph types?
def visualise_results(eval_dict):
    # Generate all the figures
    create_bar_chart_all_datasets_all_classes(eval_dict)
    # create_group_3_bar_chart_all_datasets(
    #                                         eval_dict,
    #                                         ("car", "bus", "truck"),
    #                                         "mAP average for large vehicles for all datasets",
    #                                         "./output_figures/all_datasets_large_vehicles.png")
    create_group_3_bar_chart_all_datasets(eval_dict)

    create_group_3_bar_chart_3_datasets(eval_dict)


# Load the config file
cfg = load_yaml(CONFIG_PATH)

# Read and filter from the result yaml files
ed = get_eval_dict(cfg)
print(ed)

# Plot some bar charts
visualise_results(ed)
