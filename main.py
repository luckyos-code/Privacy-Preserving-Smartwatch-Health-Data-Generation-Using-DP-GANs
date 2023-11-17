import os
import json
import itertools

from run_experiment import ci_experiment
from stress_slurm import config

# fix annoying harmless warnings - sadly not working #TODO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def check_create_folder(dir: str):
    """Check if a folder exists on the current file, if not, this function creates that folder."""
    base_dir = os.path.realpath(os.getcwd())
    check_dir = os.path.join(base_dir, dir)
    if not os.path.exists(check_dir):
        print(f"Directory {check_dir} does not exist, creating it")
        os.makedirs(check_dir)

def save_dict_as_json( #TODO make cool result dataframe and save this too
    data: dict,
    path: str,
    file_name: str = None
):
    check_create_folder(path)
    if file_name is not None:
        filename = os.path.join(path,
                                f"{file_name}.json")
    else:
        filename = os.path.join(path, "run.json")

    print(f"Saving run to: {filename}")
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def scenario_run(scenario_id: int, saving: bool = False):
    # -scenarios:
    #   -non-private for both cnn and transformer
    #       -real LOSO = 1
    #       -best GAN for different numbers in LOSO = 4
    #       -best GAN for different numbers in TSTR = 3
    #   -private for both cnn and transformer
    #       -real LOSO = 1
    #       -privacy gans for different numbers in TSTR = 3
    #
    # TODO
    # privatize gan afterwards
    # private gan and real data privatized
    # gan and real data privatized
    # validation split 0.2 in TSTR

    base_save_folder = config.RESULTS_FOLDER_PATH + "/scenarios"
    run_name = ""

    # hardcoded options #TODO maybe put in config or derive from configs or something
    num_ci_runs = 10 #TODO
    models = ["CNN-LSTM", "CNN", "Transformer"]
    np_gan_models = ["TIMEGAN", "DGAN", "CGAN"]
    p_gan_models = ["CGAN", "DPCGAN-e-10", "DPCGAN-e-1", "DPCGAN-e-0.1"]
    syn_subjs = [1, 5, 10, 15, 30, 50, 100]
    eps_values = [None, 10, 1, 0.1]
    eval_modes = ["LOSO", "TSTR"]

    # 1 - LOSO - real data - all models - all eps
    if scenario_id == 1:
        scenario_str = "LOSO_15real"
        scenario_name = f"{scenario_id}-{scenario_str}"
        save_folder = base_save_folder + "/" + scenario_name
        print(f"***Running scenario {scenario_id}: {scenario_name}")

        iter_lst = list(itertools.product(models, eps_values))
        exp_num = len(iter_lst)
        for i, (model, eps) in enumerate(iter_lst):
            run_name = f"{model}_{scenario_str}"
            run_name += "" if not eps else f"_eps{str(eps)}"
            print(f"**Starting experiment run ({i+1}/{exp_num}): {run_name}...")
            run_dict = ci_experiment(
                num_runs=num_ci_runs,
                real_subj_cnt=15, #TODO
                syn_subj_cnt=0,
                gan_mode=None,
                sliding_windows=False,
                eval_mode="LOSO",
                nn_mode=model,
                eps=eps,
                silent_runs=True #TODO
            )
            if saving:
                save_dict_as_json(run_dict, path=save_folder, file_name=run_name)
            print("")
    # 2 - TSTR - all GANs - 15 subj - no eps
    elif scenario_id == 2:
        scenario_str = "TSTR_15syn"
        scenario_name = f"{scenario_id}-{scenario_str}"
        save_folder = base_save_folder + "/" + scenario_name
        print(f"***Running scenario {scenario_id}: {scenario_name}")

        iter_lst = list(itertools.product(models, np_gan_models))
        exp_num = len(iter_lst)
        for i, (model, gan_mode) in enumerate(iter_lst):
            run_name = f"{model}_{scenario_str}_{gan_mode}"
            print(f"**Starting experiment run ({i+1}/{exp_num}): {run_name}...")
            run_dict = ci_experiment(
                num_runs=num_ci_runs,
                real_subj_cnt=15,
                syn_subj_cnt=15,
                gan_mode=gan_mode,
                sliding_windows=False,
                eval_mode="TSTR",
                nn_mode=model,
                eps=None,
                silent_runs=True
            )
            if saving:
                save_dict_as_json(run_dict, path=save_folder, file_name=run_name)
            print("")
    # 3 - TSTR - CGAN and DP-CGAN - different subj counts - training no eps
    elif scenario_id == 3:
        scenario_str = "TSTR_cGAN"
        scenario_name = f"{scenario_id}-{scenario_str}"
        save_folder = base_save_folder + "/" + scenario_name
        print(f"***Running scenario {scenario_id}: {scenario_name}")

        iter_lst = list(itertools.product(models, p_gan_models, syn_subjs))
        exp_num = len(iter_lst)
        for i, (model, gan_mode, syn_subj_cnt) in enumerate(iter_lst):
            run_name = f"{model}_{scenario_str}_syn{syn_subj_cnt}_{gan_mode}"
            print(f"**Starting experiment run ({i+1}/{exp_num}): {run_name}...")
            run_dict = ci_experiment(
                num_runs=num_ci_runs,
                real_subj_cnt=15,
                syn_subj_cnt=syn_subj_cnt,
                gan_mode=gan_mode,
                sliding_windows=False,
                eval_mode="TSTR",
                nn_mode=model,
                eps=None,
                silent_runs=True
            )
            if saving:
                save_dict_as_json(run_dict, path=save_folder, file_name=run_name)
            print("")
    # 4 - LOSO - CGAN - different subj counts - no eps
    elif scenario_id == 4:
        scenario_str = "LOSO_cGAN"
        scenario_name = f"{scenario_id}-{scenario_str}"
        save_folder = base_save_folder + "/" + scenario_name
        print(f"***Running scenario {scenario_id}: {scenario_name}")

        iter_lst = list(itertools.product(models, syn_subjs))
        exp_num = len(iter_lst)
        for i, (model, syn_subj_cnt) in enumerate(iter_lst):
            run_name = f"{model}_{scenario_str}_{syn_subj_cnt}"
            print(f"**Starting experiment run ({i+1}/{exp_num}): {run_name}...")
            run_dict = ci_experiment(
                num_runs=num_ci_runs,
                real_subj_cnt=15,
                syn_subj_cnt=syn_subj_cnt,
                gan_mode="CGAN",
                sliding_windows=False,
                eval_mode="LOSO",
                nn_mode=model,
                eps=None,
                silent_runs=True
            )
            if saving:
                save_dict_as_json(run_dict, path=save_folder, file_name=run_name)
            print("")
    # 5 - noised data evaluation (no official dp)
    elif scenario_id == 5:
        noise_options = [i for i in range(1,10)]
        scenario_str = "LOSO_15real_noised"
        scenario_name = f"{scenario_id}-{scenario_str}"
        save_folder = base_save_folder + "/" + scenario_name
        print(f"***Running scenario {scenario_id}: {scenario_name}")

        iter_lst = list(itertools.product(["CNN"], noise_options)) # TODO
        exp_num = len(iter_lst)
        for i, (model, data_noise_parameter) in enumerate(iter_lst):
            run_name = f"{model}_{scenario_str}{data_noise_parameter}"
            print(f"**Starting experiment run ({i+1}/{exp_num}): {run_name}...")
            run_dict = ci_experiment(
                num_runs=num_ci_runs,
                real_subj_cnt=15,
                syn_subj_cnt=0,
                gan_mode=None,
                sliding_windows=False,
                eval_mode="LOSO",
                nn_mode=model,
                eps=None,
                silent_runs=True,
                data_noise_parameter=data_noise_parameter,
            )
            if saving:
                save_dict_as_json(run_dict, path=save_folder, file_name=run_name)
            print("")

# get inputs and run experiment with these settings
def main():
    parser = config.create_arg_parse_instance()
    args = parser.parse_args()

    if not args.id:
        print(f"Selected args:\n{args}")
        run_dict = ci_experiment(
            num_runs=args.runs,
            real_subj_cnt=args.real,
            syn_subj_cnt=args.syn,
            gan_mode=args.gan,
            sliding_windows=args.sliding,
            eval_mode=args.eval,
            nn_mode=args.model,
            eps=args.privacy if args.privacy != 0 else None,
            silent_runs=True
        )
        if args.saving:
            save_dict_as_json(run_dict, path=config.RESULTS_FOLDER_PATH, file_name="args_run")
    else:
        scenario_run(args.id, args.saving)


if __name__ == "__main__":
    main()