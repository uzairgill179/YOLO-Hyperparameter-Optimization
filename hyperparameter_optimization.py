"""
hyperparameter_optimization.py
================================
Hyperparameter optimization for YOLO11-based object detection using a
hybrid Differential Evolution (DE) + Bayesian Optimization (BO) strategy.

Published as supplementary material for:
    [Your paper title here]
    [Journal of Scientific Reports, Year]

Authors : [Your Name(s)]
License : MIT (see LICENSE)

Usage
-----
    python hyperparameter_optimization.py \
        --data_yaml        /path/to/dataset/data.yaml \
        --project_dir      /path/to/output/directory \
        --model_yaml       /path/to/yolo11n.yaml \
        --pretrained_weights /path/to/yolo11n.pt \
        --generations      10 \
        --population_size  5 \
        --bo_iterations    15 \
        --optimizers       Adam

Requirements
------------
    pip install ultralytics scikit-optimize torch numpy matplotlib pyyaml
"""

import os
import random
import json
import logging
import pickle
import time
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args


# ==============================================================================
# Argument Parsing
# ==============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid Differential Evolution + Bayesian Optimization for "
            "YOLO11 hyperparameter tuning."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data / paths ---
    parser.add_argument(
        "--data_yaml",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the dataset YAML file (e.g. /path/to/dataset/data.yaml). "
             "The file must follow the Ultralytics dataset format with 'train', "
             "'val', and 'nc' keys.",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        required=True,
        metavar="PATH",
        help="Root directory where all experiment results will be saved.",
    )
    parser.add_argument(
        "--model_yaml",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the YOLO11 architecture YAML file "
             "(e.g. /path/to/ultralytics/cfg/models/11/yolo11n.yaml).",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional path to pretrained YOLO11 weights (.pt file). "
             "If omitted, the model trains from scratch.",
    )

    # --- DE settings ---
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations for Differential Evolution.",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=5,
        help="Population size for Differential Evolution.",
    )

    # --- BO settings ---
    parser.add_argument(
        "--bo_iterations",
        type=int,
        default=15,
        help="Number of evaluations for Bayesian Optimization.",
    )

    # --- Optimizer selection ---
    parser.add_argument(
        "--optimizers",
        type=str,
        nargs="+",
        choices=["SGD", "Adam", "both"],
        default=["Adam"],
        help="Gradient-descent optimizer(s) to evaluate: SGD, Adam, or both.",
    )

    return parser


# ==============================================================================
# Reproducibility
# ==============================================================================
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

# ==============================================================================
# Device
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Hyperparameter Search Space
# ==============================================================================
HYPERPARAMETER_SPACE = {
    "learning_rate": [1e-4, 1e-2],   # continuous – [min, max]
    "batch_size":    [8,    16],      # integer    – [min, max]
    "momentum":      [0.85, 0.95],    # continuous – [min, max]
    "weight_decay":  [1e-5, 1e-3],    # continuous – [min, max]
    "optimizer":     ["SGD", "Adam"], # categorical
}

# ==============================================================================
# Fidelity Levels  (low = quick screening; high = full training)
# ==============================================================================
FIDELITY_LEVELS = {
    "low":  {"epochs": 5},
    "high": {"epochs": 500},
}

# ==============================================================================
# Early-Stopping Patience
# ==============================================================================
OPTIMIZATION_PATIENCE = 5    # generations without improvement → stop DE
TRAINING_PATIENCE     = 100  # epochs without validation gain  → stop YOLO train

# ==============================================================================
# Differential Evolution Control Parameters
# ==============================================================================
F_MIN, F_MAX = 0.4, 0.9   # mutation scale factor range (adaptive)
CR = 0.9                   # crossover probability


# ==============================================================================
# Logging
# ==============================================================================
def setup_logging(log_path: str) -> None:
    """Configure root logger to write to *log_path*."""
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


# ==============================================================================
# Per-Optimizer Environment Setup
# ==============================================================================
def setup_optimizer_environment(project_dir: str, optimizer_name: str):
    """
    Create the output directory for *optimizer_name* and return its path
    together with the path to the iteration log file.
    """
    optimizer_dir = os.path.join(project_dir, f"{optimizer_name}_optimizer")
    os.makedirs(optimizer_dir, exist_ok=True)

    log_file = os.path.join(optimizer_dir, "iteration_log.txt")

    # Write header only when starting fresh (no existing checkpoint)
    checkpoint_file = os.path.join(optimizer_dir, "checkpoint.pkl")
    if not os.path.exists(checkpoint_file):
        with open(log_file, "w") as fh:
            fh.write("Generation, Individual Index, Hyperparameters, Fitness\n")

    return optimizer_dir, log_file


# ==============================================================================
# Checkpointing
# ==============================================================================
def save_checkpoint(
    optimizer_dir: str,
    population: list,
    history: list,
    generation: int,
    best_fitness: float,
    best_hyperparams: dict,
    epochs_no_improve: int,
    phase: str = "DE",
) -> None:
    """Persist optimisation state to disk so a run can be resumed."""
    checkpoint_file = os.path.join(optimizer_dir, "checkpoint.pkl")
    data = {
        "population":      population,
        "history":         history,
        "generation":      generation,
        "best_fitness":    best_fitness,
        "best_hyperparams": best_hyperparams,
        "epochs_no_improve": epochs_no_improve,
        "phase":           phase,
    }
    with open(checkpoint_file, "wb") as fh:
        pickle.dump(data, fh)

    msg = f"Checkpoint saved – generation {generation}, phase {phase}"
    logging.info(msg)
    print(msg)


def load_checkpoint(optimizer_dir: str):
    """
    Load and return a checkpoint dict if one exists, otherwise return *None*.
    """
    checkpoint_file = os.path.join(optimizer_dir, "checkpoint.pkl")
    if not os.path.exists(checkpoint_file):
        return None

    with open(checkpoint_file, "rb") as fh:
        data = pickle.load(fh)

    msg = (
        f"Resuming from checkpoint – "
        f"generation {data['generation']}, phase {data['phase']}"
    )
    logging.info(msg)
    print(msg)
    return data


# ==============================================================================
# DE: Population Initialisation
# ==============================================================================
def initialize_population(hyperparameter_space: dict, population_size: int) -> list:
    """
    Randomly sample *population_size* individuals from *hyperparameter_space*.

    Parameters
    ----------
    hyperparameter_space : dict
        Keys map to either [min, max] (numeric) or a list of choices (categorical).
    population_size : int

    Returns
    -------
    list of dict
    """
    population = []
    for _ in range(population_size):
        individual = {
            "learning_rate": float(random.uniform(*hyperparameter_space["learning_rate"])),
            "batch_size":    int(random.uniform(*hyperparameter_space["batch_size"])),
            "momentum":      float(random.uniform(*hyperparameter_space["momentum"])),
            "weight_decay":  float(random.uniform(*hyperparameter_space["weight_decay"])),
            "optimizer":     str(random.choice(hyperparameter_space["optimizer"])),
        }
        population.append(individual)
    return population


# ==============================================================================
# DE: Mutation + Crossover
# ==============================================================================
def mutate_and_crossover(
    population: list,
    target_idx: int,
    generation: int,
    max_generations: int,
    hyperparameter_space: dict,
) -> dict:
    """
    Produce a trial vector for *population[target_idx]* via DE/rand/1/bin mutation
    with an adaptive mutation scale factor F.

    Parameters
    ----------
    population        : current population
    target_idx        : index of the target individual
    generation        : current generation (0-based)
    max_generations   : total number of planned generations
    hyperparameter_space : search-space bounds / choices

    Returns
    -------
    trial : dict  – the trial individual
    """
    # Adaptive F: decreases linearly from F_MAX to F_MIN
    F = F_MIN + (F_MAX - F_MIN) * (1 - generation / max_generations)

    candidate_idxs = [i for i in range(len(population)) if i != target_idx]
    a, b, c = (population[random.choice(candidate_idxs)] for _ in range(3))
    target   = population[target_idx]

    # --- Mutation ---
    mutant = {}
    for key in hyperparameter_space:
        if key == "optimizer":
            mutant[key] = random.choice(hyperparameter_space["optimizer"])
        else:
            value = a[key] + F * (b[key] - c[key])
            lo, hi = hyperparameter_space[key]
            mutant[key] = float(np.clip(value, lo, hi))

    # --- Binomial crossover ---
    trial = {}
    for key in hyperparameter_space:
        trial[key] = mutant[key] if random.random() <= CR else target[key]

    return trial


# ==============================================================================
# Fitness Evaluation
# ==============================================================================
def evaluate_hyperparameters(
    hyperparams: dict,
    fidelity: str,
    data_yaml: str,
    project_dir: str,
    generation,
    individual_index,
    model_yaml: str,
    pretrained_weights: str,
) -> tuple:
    """
    Train a YOLO11 model with *hyperparams* and return (fitness, metrics).

    Fitness is defined as the negative harmonic mean of Precision, Recall,
    and mAP@[0.5:0.95], so *lower* is better (consistent with minimisation).

    Parameters
    ----------
    hyperparams        : dict with keys learning_rate, batch_size, momentum,
                         weight_decay, optimizer
    fidelity           : 'low' (quick screen) or 'high' (full training)
    data_yaml          : path to the dataset YAML
    project_dir        : directory for saving this run's outputs
    generation         : generation label (int or string) for naming
    individual_index   : individual label for naming
    model_yaml         : path to the YOLO11 architecture YAML
    pretrained_weights : path to pretrained .pt file, or None

    Returns
    -------
    fitness  : float  (negative harmonic mean; lower = better)
    metrics  : dict   {'precision', 'recall', 'mAP'}
    """
    start_time = time.time()
    try:
        lr           = float(hyperparams["learning_rate"])
        batch_size   = int(hyperparams["batch_size"])
        momentum     = float(hyperparams["momentum"])
        weight_decay = float(hyperparams["weight_decay"])
        optimizer_name = str(hyperparams["optimizer"])
        epochs       = FIDELITY_LEVELS[fidelity]["epochs"]

        # --- Build model ---
        logging.info(f"Initialising YOLO model from: {model_yaml}")
        model = YOLO(model_yaml)

        if pretrained_weights and os.path.exists(pretrained_weights):
            logging.info(f"Loading pretrained weights: {pretrained_weights}")
            model.load(pretrained_weights)

        torch.cuda.empty_cache()

        exp_name = f"gen_{generation}_ind_{individual_index}_{fidelity}"

        # --- Train ---
        model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            lr0=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            optimizer=optimizer_name.lower(),
            device=DEVICE,
            project=project_dir,
            name=exp_name,
            exist_ok=False,
            workers=0,
            cache=False,
            pretrained=True,
            patience=TRAINING_PATIENCE,
            verbose=False,
        )

        # --- Validate ---
        val_metrics   = model.val(data=data_yaml, batch=batch_size, imgsz=640)
        avg_precision = float(val_metrics.box.p.mean())
        avg_recall    = float(val_metrics.box.r.mean())
        avg_mAP       = float(val_metrics.box.map)

        # Harmonic mean of P, R, mAP (negated for minimisation)
        fitness = -(3.0 / (1.0 / avg_precision + 1.0 / avg_recall + 1.0 / avg_mAP))

        elapsed = time.time() - start_time
        logging.info(
            f"Gen={generation}, Ind={individual_index}, Fidelity={fidelity} | "
            f"Fitness={-fitness:.4f}, P={avg_precision:.4f}, "
            f"R={avg_recall:.4f}, mAP={avg_mAP:.4f}, Time={elapsed:.1f}s"
        )

        # Remove bulky weight files after low-fidelity screening to save disk space
        if fidelity == "low":
            weights_dir = os.path.join(project_dir, exp_name, "weights")
            for fname in ("best.pt", "last.pt"):
                fpath = os.path.join(weights_dir, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
                    logging.info(f"Deleted low-fidelity weight: {fpath}")

        return fitness, {"precision": avg_precision, "recall": avg_recall, "mAP": avg_mAP}

    except Exception as exc:
        logging.error(f"Evaluation failed for {hyperparams}: {exc}")
        return float("inf"), {"precision": 0.0, "recall": 0.0, "mAP": 0.0}


# ==============================================================================
# Hybrid DE + BO
# ==============================================================================
def differential_evolution_with_bayesian_optimization(
    data_yaml: str,
    project_dir: str,
    iteration_log_file: str,
    hyperparameter_space: dict,
    generations: int,
    population_size: int,
    bo_iterations: int,
    model_yaml: str,
    pretrained_weights: str,
):
    """
    Run the full DE → high-fidelity evaluation → BO pipeline.

    Phases
    ------
    1. Differential Evolution (low-fidelity screening per generation)
    2. High-fidelity evaluation of the DE best candidate
    3. Bayesian Optimisation (high-fidelity, warm-started from DE best)

    Returns
    -------
    bo_best  : dict  – best hyperparameter set found by BO
    history  : list  – list of all evaluated individuals with fitness/metrics
    """

    # ---- Helper bound to this call's paths / settings ----
    def _evaluate(hyperparams, fidelity, gen_label, ind_label):
        return evaluate_hyperparameters(
            hyperparams, fidelity, data_yaml, project_dir,
            gen_label, ind_label, model_yaml, pretrained_weights,
        )

    # ------------------------------------------------------------------
    # Resume or start fresh
    # ------------------------------------------------------------------
    checkpoint = load_checkpoint(project_dir)

    if checkpoint is not None:
        population       = checkpoint["population"]
        history          = checkpoint["history"]
        start_generation = checkpoint["generation"]
        best_fitness     = checkpoint["best_fitness"]
        best_hyperparams = checkpoint["best_hyperparams"]
        epochs_no_improve = checkpoint["epochs_no_improve"]
        phase            = checkpoint["phase"]

        if phase in ("BO", "HF_complete"):
            logging.info(f"Resuming from {phase} phase – skipping DE.")
            start_generation = generations  # skip DE loop
        elif phase == "Complete":
            logging.info("Optimisation already complete. Returning stored results.")
            return best_hyperparams, history
        else:
            logging.info(f"Resuming DE from generation {start_generation}.")
    else:
        population        = initialize_population(hyperparameter_space, population_size)
        history           = []
        start_generation  = 0
        best_fitness      = float("inf")
        best_hyperparams  = None
        epochs_no_improve = 0

        # Evaluate initial population (low fidelity)
        for idx, individual in enumerate(population):
            fitness, _ = _evaluate(individual, "low", 0, idx)
            history.append({"generation": 0, "individual": individual, "fitness": fitness})

        save_checkpoint(
            project_dir, population, history, 0,
            best_fitness, best_hyperparams, epochs_no_improve, phase="Initial",
        )

    # ------------------------------------------------------------------
    # Differential Evolution
    # ------------------------------------------------------------------
    if start_generation < generations:
        for generation in range(start_generation, generations):
            logging.info(f"DE Generation {generation + 1}/{generations} ...")
            new_population = []

            for idx, individual in enumerate(population):
                trial = mutate_and_crossover(
                    population, idx, generation, generations, hyperparameter_space
                )

                target_fitness, _ = _evaluate(individual, "low", generation + 1, idx)
                trial_fitness,  _ = _evaluate(trial,      "low", generation + 1, idx)

                if trial_fitness < target_fitness:
                    new_population.append(trial)
                    chosen, chosen_fitness = trial,      trial_fitness
                else:
                    new_population.append(individual)
                    chosen, chosen_fitness = individual, target_fitness

                history.append({
                    "generation": generation + 1,
                    "individual": chosen,
                    "fitness":    chosen_fitness,
                })

                # Serialise safely (numpy scalar → Python scalar)
                safe_chosen = {
                    k: (float(v) if isinstance(v, (np.float32, np.float64)) else
                        int(v)   if isinstance(v, (np.int32,  np.int64))   else v)
                    for k, v in chosen.items()
                }
                with open(iteration_log_file, "a") as fh:
                    fh.write(
                        f"{generation + 1}, {idx + 1}, "
                        f"{json.dumps(safe_chosen)}, {chosen_fitness:.4f}\n"
                    )

            population = new_population

            # Track improvement
            gen_best       = min(history[-population_size:], key=lambda x: x["fitness"])
            gen_best_fit   = gen_best["fitness"]
            gen_best_hp    = gen_best["individual"]

            if gen_best_fit < best_fitness - 1e-4:
                best_fitness     = gen_best_fit
                best_hyperparams = gen_best_hp
                epochs_no_improve = 0
                logging.info(f"New best fitness = {-best_fitness:.4f}")
            else:
                epochs_no_improve += 1

            save_checkpoint(
                project_dir, population, history, generation + 1,
                best_fitness, best_hyperparams, epochs_no_improve, phase="DE",
            )

            if epochs_no_improve >= OPTIMIZATION_PATIENCE:
                logging.info(
                    f"Early stopping DE at generation {generation + 1} "
                    "(no improvement for {OPTIMIZATION_PATIENCE} generations)."
                )
                break

    # ------------------------------------------------------------------
    # High-Fidelity Evaluation of DE Best
    # ------------------------------------------------------------------
    if best_hyperparams is not None:
        hf_dir = os.path.join(project_dir, "high_fidelity_evaluation")
        os.makedirs(hf_dir, exist_ok=True)

        logging.info("High-fidelity evaluation of best DE candidate ...")
        hf_fitness, hf_metrics = _evaluate(best_hyperparams, "high", "HF", "best")
        history.append({
            "generation": "HF",
            "individual": best_hyperparams,
            "fitness":    hf_fitness,
            "metrics":    hf_metrics,
        })
        logging.info(f"HF result: Fitness={-hf_fitness:.4f}, Metrics={hf_metrics}")

        save_checkpoint(
            project_dir, population, history, generations,
            best_fitness, best_hyperparams, epochs_no_improve, phase="HF_complete",
        )
    else:
        logging.warning("No best hyperparams found for HF evaluation.")

    # ------------------------------------------------------------------
    # Bayesian Optimisation
    # ------------------------------------------------------------------
    logging.info("Starting Bayesian Optimisation ...")

    bo_space = [
        Real(    *hyperparameter_space["learning_rate"], name="learning_rate"),
        Integer( *hyperparameter_space["batch_size"],    name="batch_size"),
        Real(    *hyperparameter_space["momentum"],      name="momentum"),
        Real(    *hyperparameter_space["weight_decay"],  name="weight_decay"),
        Categorical(hyperparameter_space["optimizer"],   name="optimizer"),
    ]

    bo_counter = [0]

    @use_named_args(bo_space)
    def _bo_objective(**params):
        idx = bo_counter[0]
        bo_counter[0] += 1

        bo_fitness, bo_metrics = _evaluate(params, "high", f"BO_{idx}", idx)
        history.append({
            "generation": f"BO_{idx}",
            "individual": params,
            "fitness":    bo_fitness,
            "metrics":    bo_metrics,
        })
        save_checkpoint(
            project_dir, population, history, generations,
            best_fitness, best_hyperparams, epochs_no_improve, phase="BO",
        )
        return bo_fitness

    # Warm-start BO from DE best (fall back to sensible defaults)
    x0 = (
        [
            best_hyperparams["learning_rate"],
            best_hyperparams["batch_size"],
            best_hyperparams["momentum"],
            best_hyperparams["weight_decay"],
            best_hyperparams["optimizer"],
        ]
        if best_hyperparams
        else [1e-3, 8, 0.9, 1e-4, "Adam"]
    )

    res = gp_minimize(
        func=_bo_objective,
        dimensions=bo_space,
        n_calls=bo_iterations,
        n_initial_points=10,
        x0=x0,
        random_state=GLOBAL_SEED,
    )

    bo_best = {
        "learning_rate": float(res.x[0]),
        "batch_size":    int(res.x[1]),
        "momentum":      float(res.x[2]),
        "weight_decay":  float(res.x[3]),
        "optimizer":     str(res.x[4]),
    }

    logging.info(f"BO best parameters: {bo_best}")
    with open(iteration_log_file, "a") as fh:
        fh.write(f"BO best params: {json.dumps(bo_best)}\n")

    save_checkpoint(
        project_dir, population, history, generations,
        best_fitness, bo_best, epochs_no_improve, phase="Complete",
    )

    return bo_best, history


# ==============================================================================
# Visualisation
# ==============================================================================
def visualize_optimization_results(history: list, project_dir: str) -> None:
    """
    Plot fitness vs. DE generation and save the figure to *project_dir*.

    Only DE generations (integer labels) are included in the plot; BO
    evaluations and HF evaluations are excluded.
    """
    generations, fitness_values = [], []
    for entry in history:
        if isinstance(entry["generation"], int):
            generations.append(entry["generation"])
            fitness_values.append(entry["fitness"])

    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_values, marker="o", linestyle="-")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (negative harmonic mean of P / R / mAP)")
    plt.title("DE Optimisation History")
    plt.grid(True)

    out_path = os.path.join(project_dir, "optimization_history.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Optimisation history saved to {out_path}")
    print(f"Optimisation history saved to {out_path}")


# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = build_arg_parser()
    args   = parser.parse_args()

    # --- Resolve optimizer list ---
    optimizers_to_run = ["Adam", "SGD"] if "both" in args.optimizers else args.optimizers

    # --- Global log at project root ---
    os.makedirs(args.project_dir, exist_ok=True)
    setup_logging(os.path.join(args.project_dir, "training.log"))

    print(f"Device            : {DEVICE}")
    print(f"Model YAML        : {args.model_yaml}")
    print(f"Pretrained weights: {args.pretrained_weights}")
    print(f"Dataset YAML      : {args.data_yaml}")
    print(f"Optimizers        : {', '.join(optimizers_to_run)}")

    for optimizer_name in optimizers_to_run:
        logging.info(f"=== Starting optimisation with {optimizer_name} ===")

        optimizer_dir, iter_log = setup_optimizer_environment(
            args.project_dir, optimizer_name
        )

        # Restrict 'optimizer' key to the current choice
        local_space = {**HYPERPARAMETER_SPACE, "optimizer": [optimizer_name]}

        bo_best, history = differential_evolution_with_bayesian_optimization(
            data_yaml=args.data_yaml,
            project_dir=optimizer_dir,
            iteration_log_file=iter_log,
            hyperparameter_space=local_space,
            generations=args.generations,
            population_size=args.population_size,
            bo_iterations=args.bo_iterations,
            model_yaml=args.model_yaml,
            pretrained_weights=args.pretrained_weights,
        )

        visualize_optimization_results(history, optimizer_dir)

        print(f"\n{'='*60}")
        print(f"Best hyperparameters found ({optimizer_name}):")
        for k, v in bo_best.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}\n")
