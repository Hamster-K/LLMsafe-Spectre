import subprocess
import datetime
import os
import time
from multiprocessing import Process

# ================== 全局参数 ==================
timeout = 60
model = "model"
sudoPassword = "Password"
attack_types = ["spectrev1", "spectrev2", "spectrev4", "other Attackes..."]
BASE_SAVE_DIR = "../Data_"
BASE_ATTACK_DIR = "../Attack"


# ================== perf  ==================
def HPC(attack_type, if_attack, output_path):
    cmd = (
        "perf stat "
        "-a -e branches,branch-misses,r4f2e,r412e,r0148,other events... "
        f"-I 100 -o {output_path}"
    )

    command = cmd.split()
    cmd1 = subprocess.Popen(['echo', sudoPassword], stdout=subprocess.PIPE)
    subprocess.Popen(['sudo', '-S'] + command, stdin=cmd1.stdout)

# ================== Spectre ==================
def spectre(attack_type):
    attack_bin = os.path.join(BASE_ATTACK_DIR, attack_type, "a.out")
    try:
        subprocess.run(attack_bin, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[{attack_type}] end")

    print(f"[{attack_type}] end")


# ====================================
def run_one_experiment(attack_type, if_attack):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = os.path.join(
        BASE_SAVE_DIR,
        attack_type,
        model,
        f"if_attack_{if_attack}"
    )
    os.makedirs(save_dir, exist_ok=True)

    file_name = f"output_{current_time}.txt"
    output_path = os.path.join(save_dir, file_name)

    print("=" * 60)
    print(f"Attack Type : {attack_type}")
    print(f"If Attack  : {if_attack}")
    print(f"Output     : {output_path}")
    print("=" * 60)

    perf_process = Process(
        target=HPC,
        args=(attack_type, if_attack, output_path)
    )

    if if_attack == 1:
        attack_process = Process(
            target=spectre,
            args=(attack_type,)
        )

    # START
    perf_process.start()
    if if_attack == 1:
        print(">>> attack")
        attack_process.start()

    time.sleep(timeout)

    print(">>> clean")

    # END
    perf_process.terminate()
    if if_attack == 1:
        attack_process.terminate()

    os.system("sudo pkill -9 perf")
    print(">>> perf end\n")


# ====================================
if __name__ == "__main__":
    try:
        for attack_type in attack_types:

            run_one_experiment(attack_type, if_attack=1)
            run_one_experiment(attack_type, if_attack=0)

    except KeyboardInterrupt:
        os.system("sudo pkill -9 perf")
        print("end")

    print("✅")