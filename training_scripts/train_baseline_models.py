import argparse
import re
import subprocess
import traceback
from pathlib import Path

import numpy as np


def run(cmd: str):
    try:
        cmd = re.sub('[ ]+', ' ', cmd)
        print(f'Starting command: {cmd} ')

        # cmd = cmd.split(' ')
        subprocess.run(cmd, check=True, shell=True, universal_newlines=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)

    except subprocess.CalledProcessError as e:
        print('error: ', e)
        print(traceback.format_exc())
        return 1

    except Exception as e:
        print('error: ', e)
        print(traceback.format_exc())
        return 1

    print(f'Finished command: {cmd}')
    return 0


def get_overrides(lr: float, bs: int, num_epochs: int, patience: int,
                  seeds: bool, power: int, warmup_steps: int, beam_size: int):
    if seeds:
        s1 = np.random.randint(10000, 20000, 1)
        s2 = s1 // 10
        s3 = s1 // 100
    else:
        s1, s2, s3 = 13370, 1337, 133
    return '"trainer":{"learning_rate_scheduler":{"warmup_steps":%d,"power":%d},"optimizer":{"lr":%f},"num_epochs":%d,"patience":%d,' \
           '"cuda_device":0},"distributed":null,' \
           '"data_loader":{"batch_size":%d},' \
           '"model":{"beam_size":%d,' \
           '"random_seed":%d,"numpy_seed":%d,"pytorch_seed":%d}' \
           % (warmup_steps, power, lr, num_epochs,
              patience, bs, beam_size, s1, s2, s3), \
           "data/program_dev.tsv"


def run_experiment(absolute_path_project_root: str, lr: float, bs: int, num_epochs: int, patience: int,
                   seeds: bool, power: int, warmup_steps: int, beam_size: int, seed: int):
    absolute_path_project_root = Path(absolute_path_project_root)
    config_path = absolute_path_project_root / "experiments" / "baseline.json"
    ser_dir = absolute_path_project_root / "experiments" / f"baseline_{seed}"

    overrides, test_evaluation_data_path = get_overrides(lr, bs, num_epochs, patience,
                                                         seeds, power, warmup_steps, beam_size)
    overrides = overrides.replace(' ', '')

    cmd = f"allennlp " \
          f"train " \
          f"{config_path} " \
          f"-s {ser_dir} " \
          f"--include-package models_code " \
          f"--overrides '{overrides}' "

    response = run(cmd)
    if not Path(f"{ser_dir}/dev_metrics.json").exists():
        evaluate_test_cmd = f"allennlp evaluate {ser_dir}/model.tar.gz " \
                            f"{test_evaluation_data_path} " \
                            f"--output-file {ser_dir}/program_dev_metrics.json " \
                            f"--cuda-device 0 " \
                            f"--include-package models_code " \
                            f"--batch-size 24"
        if response == 0:
            run(evaluate_test_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--absolute-path-project-root', type=str, help="absolute path to the root of the project")
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--seeds', dest='seeds', action='store_true')
    parser.add_argument('--seed-num', type=int, default=0)
    parser.add_argument('--ws', type=int, default=1500)
    parser.add_argument('--power', type=int, default=1)
    parser.add_argument('--beam', type=int, default=4)

    args = parser.parse_args()
    seed = np.random.randint(100)
    np.random.seed(seed)

    run_experiment(args.absolute_path_project_root, args.lr, args.bs, args.num_epochs, args.patience,
                   args.seeds, args.power, args.warmup_steps, args.beam_size, args.seed_num)
