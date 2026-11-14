# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

# Modified from github.com/facebookresearch/meru

"""
Train a HyCoCLIP, PHyCLIP, MERU, SigMERU, or CLIP model based on a config file.
"""

import argparse
import multiprocessing as mp
import random
import socket
import sys
import time
from pathlib import Path

if mp.get_start_method(allow_none=True) != "spawn":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.amp.grad_scaler import GradScaler

import sigmeru.utils.distributed as dist
from sigmeru.config import LazyConfig, LazyFactory
from sigmeru.models import HyCoCLIP, PHyCLIP
from sigmeru.tokenizer import Tokenizer
from sigmeru.utils.checkpointing import CheckpointManager
from sigmeru.utils.directory import generate_output_dir_name
from sigmeru.utils.timer import Timer
from sigmeru.utils.worker import evaluation_worker, tensorboard_worker

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("--config", help="Path to a .py config file.")
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume from last_checkpoint.txt in the generated output directory.",
)
parser.add_argument(
    "--resume-from",
    default="",
    help=(
        "Optional explicit checkpoint file or experiment directory to resume from. "
        "If a directory is given, last_checkpoint.txt inside it is used. This "
        "takes precedence over --resume."
    ),
)
parser.add_argument(
    "--target-iteration",
    type=int,
    default=None,
    help=(
        "Absolute iteration to train to. If omitted, train.num_iterations from the "
        "config/overrides is used. Example: resume from iter 15000 and train to "
        "--target-iteration 500000."
    ),
)
parser.add_argument(
    "--checkpoint-period", type=int, default=50000, help="Checkpoint saving period."
)
parser.add_argument(
    "--log-period",
    type=int,
    default=100,
    help="Log to stdout/tensorboard periodically on the main process.",
)
parser.add_argument(
    "--num-machines",
    type=int,
    default=1,
    help="Number of machines used in distributed training.",
)
parser.add_argument(
    "--num-gpus", type=int, default=0, help="Number of GPUs per machine."
)
parser.add_argument(
    "--machine-rank",
    type=int,
    default=0,
    help="Integer in [0, num_machines) specifying machine ID.",
)
_random_port = random.randint(2000, 19999)
parser.add_argument(
    "--dist-url",
    default=f"tcp://127.0.0.1:{_random_port}",
    help="URL of the main process in distributed training.",
)
parser.add_argument(
    "overrides", nargs="...", default=[], help="Config overrides as dotlist pairs."
)


def process_logging_value(log_data, prefix, name, value):
    """Process a scalar or list of scalars for TensorBoard logging."""
    if isinstance(value, list):
        for i, v in enumerate(value):
            key = f"{prefix}/{name}_{i:03d}"
            if torch.is_tensor(v):
                scalar_value = v.detach().cpu().item()
            else:
                scalar_value = float(v)
            log_data["scalar"][key] = scalar_value
    else:
        key = f"{prefix}/{name}"
        if torch.is_tensor(value):
            scalar_value = value.detach().cpu().item()
        else:
            scalar_value = float(value)
        log_data["scalar"][key] = scalar_value


def _model_uses_explicit_box_forward(model: torch.nn.Module) -> bool:
    actual_model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    return isinstance(actual_model, (HyCoCLIP, PHyCLIP))


def _model_uses_box_augmentation(model: torch.nn.Module) -> bool:
    actual_model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    if isinstance(actual_model, (HyCoCLIP, PHyCLIP)):
        return False
    return bool(getattr(actual_model, "use_boxes", False))


def _move_tensor_to_device(
    tensor: torch.Tensor,
    device: torch.device,
    non_blocking: bool,
) -> torch.Tensor:
    """Move a tensor to the target device with optional non-blocking H2D copy."""
    return tensor.to(
        device,
        non_blocking=bool(non_blocking and device.type == "cuda"),
    )


def _configure_tf32(cfg) -> None:
    """Configure TensorFloat-32 controls from the train config."""
    use_tf32 = bool(getattr(cfg.train, "use_tf32", False))
    matmul_precision = str(getattr(cfg.train, "float32_matmul_precision", "high"))
    cudnn_tf32 = bool(getattr(cfg.train, "cudnn_tf32", use_tf32))
    cuda_matmul_tf32 = bool(getattr(cfg.train, "cuda_matmul_tf32", use_tf32))

    if torch.cuda.is_available():
        if use_tf32:
            torch.set_float32_matmul_precision(matmul_precision)
        else:
            torch.set_float32_matmul_precision("highest")

        torch.backends.cuda.matmul.allow_tf32 = bool(cuda_matmul_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cudnn_tf32)


def _tf32_status_string(cfg) -> str:
    return (
        f"use_tf32={bool(getattr(cfg.train, 'use_tf32', False))}, "
        f"float32_matmul_precision={getattr(cfg.train, 'float32_matmul_precision', 'high')}, "
        f"cuda.matmul.allow_tf32={getattr(torch.backends.cuda.matmul, 'allow_tf32', 'N/A')}, "
        f"cudnn.allow_tf32={getattr(torch.backends.cudnn, 'allow_tf32', 'N/A')}"
    )


def _resume_output_dir_from_arg(resume_from: str) -> Path | None:
    """Return experiment directory implied by --resume-from, if provided."""
    if not resume_from:
        return None
    path = Path(resume_from).expanduser()
    if path.is_dir():
        return path
    # If path does not exist yet, still interpret a .pth suffix as a checkpoint
    # file path and return its parent so the subsequent load can report the true
    # file-not-found error.
    if path.suffix == ".pth":
        return path.parent
    return path


def _load_config_for_run(_A: argparse.Namespace):
    """Load config, apply overrides, and decide the output directory.

    Resume behavior:
      * --resume-from /path/to/exp_dir: use exp_dir/config.yaml if present and
        resume exp_dir/last_checkpoint.txt.
      * --resume-from /path/to/checkpoint_XXXXXXXX.pth: use checkpoint parent as
        output dir and parent/config.yaml if present.
      * --resume: use the generated output dir and its config.yaml if present.

    In all cases CLI dotlist overrides are applied last, and --target-iteration
    is applied after those overrides.
    """
    cfg = LazyConfig.load(_A.config)
    cfg = LazyConfig.apply_overrides(cfg, _A.overrides)

    explicit_resume_dir = _resume_output_dir_from_arg(_A.resume_from)
    if explicit_resume_dir is not None:
        output_dir = explicit_resume_dir
        saved_config_yaml_path = output_dir / "config.yaml"
        if saved_config_yaml_path.exists():
            logger.info(
                f"Resuming from explicit path. Loading saved config from {saved_config_yaml_path}"
            )
            cfg = LazyConfig.load(str(saved_config_yaml_path))
            if _A.overrides:
                cfg = LazyConfig.apply_overrides(cfg, _A.overrides)
        else:
            logger.warning(
                f"No config.yaml found in {output_dir}; using the command-line config."
            )
    else:
        output_dir = Path(generate_output_dir_name(cfg))
        saved_config_yaml_path = output_dir / "config.yaml"
        if _A.resume and saved_config_yaml_path.exists():
            logger.info(
                f"Resuming training. Loading saved config from {saved_config_yaml_path}"
            )
            try:
                cfg = LazyConfig.load(str(saved_config_yaml_path))
                logger.info("Loaded saved YAML config successfully")
            except Exception as e:
                logger.warning(f"Failed to load YAML config: {e}")
                logger.info("Using current config file")
            if _A.overrides:
                cfg = LazyConfig.apply_overrides(cfg, _A.overrides)
        elif _A.resume:
            logger.info("No saved config found, using current config file")

    # Add an optional experiment tag to the output directory.  This prevents
    # different loss/margin ablations from appending logs/checkpoints to the
    # same generated directory.  Explicit --resume-from keeps its directory.
    experiment_tag = str(getattr(cfg.train, "experiment_tag", "") or "").strip()
    if experiment_tag and explicit_resume_dir is None:
        safe_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in experiment_tag)
        if safe_tag and not output_dir.name.endswith(f"_{safe_tag}"):
            output_dir = output_dir.parent / f"{output_dir.name}_{safe_tag}"

    if _A.target_iteration is not None:
        if _A.target_iteration < 0:
            raise ValueError("--target-iteration must be non-negative.")
        cfg.train.num_iterations = int(_A.target_iteration)

    return cfg, output_dir


def main(_A: argparse.Namespace):
    # -------------------------------------------------------------------------
    #   BASIC SETUP FOR TRAINING JOB.
    # -------------------------------------------------------------------------
    _C, output_dir = _load_config_for_run(_A)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_eval = bool(getattr(_C.train, "run_eval", True))
    non_blocking_to_device = bool(getattr(_C.train, "non_blocking_to_device", True))
    zero_grad_set_to_none = bool(getattr(_C.train, "zero_grad_set_to_none", True))
    sync_cuda_timing = bool(getattr(_C.train, "sync_cuda_timing", False))
    log_all_ranks_to_stderr = bool(getattr(_C.train, "log_all_ranks_to_stderr", False))

    tensorboard_input_queue = None
    tensorboard_process = None
    evaluation_input_queue = None
    evaluation_process = None

    if dist.is_main_process():
        tensorboard_input_queue = mp.Queue()
        tensorboard_process = mp.Process(
            target=tensorboard_worker,
            args=(tensorboard_input_queue, output_dir),
        )
        tensorboard_process.start()

        if run_eval:
            evaluation_input_queue = mp.Queue()
            evaluation_process = mp.Process(
                target=evaluation_worker,
                args=(evaluation_input_queue, tensorboard_input_queue),
            )
            evaluation_process.start()

    def _shutdown_background_workers() -> None:
        if dist.is_main_process():
            if tensorboard_input_queue is not None:
                tensorboard_input_queue.put(None)
            if tensorboard_process is not None:
                tensorboard_process.join()
            if run_eval and evaluation_input_queue is not None and evaluation_process is not None:
                evaluation_input_queue.put(None)
                evaluation_process.join()

    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()

    if getattr(_C.train, "seed", None) is None:
        _C.train.seed = int(time.time())

    random.seed(_C.train.seed + RANK)
    np.random.seed(_C.train.seed + RANK)
    torch.manual_seed(_C.train.seed + RANK)
    torch.backends.cudnn.deterministic = _C.train.cudnn_deterministic
    torch.backends.cudnn.benchmark = _C.train.cudnn_benchmark
    _configure_tf32(_C)

    LazyConfig.save(_C, str(output_dir / "config.yaml"))

    # Keep one log file per rank, but avoid duplicated terminal logs from every
    # GPU process by default. This does not change training; it only makes the
    # console ETA readable. Set train.log_all_ranks_to_stderr=true to restore
    # old all-rank terminal logging.
    logger.remove()
    logger.add(output_dir / f"log-rank{RANK}.txt", format="{time} {level} {message}")
    if dist.is_main_process() or log_all_ranks_to_stderr:
        logger.add(sys.stderr, format="{time} | {level:<8} | {name}:{function}:{line} - {message}")

    logger.info(f"Rank of current process: {RANK}. World size: {WORLD_SIZE}")
    logger.info(f"RANK {RANK} using random seed: {_C.train.seed + RANK}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target iteration: {_C.train.num_iterations}")
    logger.info(f"TF32 config: {_tf32_status_string(_C)}")
    logger.info(OmegaConf.to_yaml(_C))

    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    # -------------------------------------------------------------------------
    #   INSTANTIATE ALL OBJECTS FOR TRAINING.
    # -------------------------------------------------------------------------
    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if _A.num_gpus != 0
        else torch.device("cpu")
    )
    dataloader = LazyFactory.build_dataloader(_C)
    tokenizer = Tokenizer()

    model = LazyFactory.build_model(_C, device)
    optimizer = LazyFactory.build_optimizer(_C, model)
    scheduler = LazyFactory.build_lr_scheduler(_C, optimizer)
    scaler = GradScaler(enabled=_C.train.amp)

    checkpoint_manager = CheckpointManager(
        output_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )

    resume_requested = bool(_A.resume or _A.resume_from)
    if _A.resume_from:
        start_iteration = checkpoint_manager.resume(_A.resume_from)
    elif _A.resume:
        start_iteration = checkpoint_manager.resume()
    else:
        start_iteration = 0

    if resume_requested:
        logger.info(
            f"Resume status: loaded iteration {start_iteration}; "
            f"will train until target iteration {_C.train.num_iterations}."
        )

    if start_iteration < 0:
        raise RuntimeError(
            "Loaded checkpoint does not contain a valid 'iteration' entry. "
            "Please resume from checkpoint_XXXXXXXX.pth, not from an old checkpoint_final.pth "
            "without iteration metadata."
        )

    if start_iteration >= int(_C.train.num_iterations):
        logger.info(
            f"Checkpoint iteration {start_iteration} is already >= target iteration "
            f"{_C.train.num_iterations}. Nothing to train."
        )
        _shutdown_background_workers()
        return

    dataloader_iter = iter(dataloader)
    timer = Timer(start_iteration + 1, total_iterations=_C.train.num_iterations)
    last_log_wall_time = time.perf_counter()
    last_log_iteration = start_iteration

    def _format_eta(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return f"{seconds // 3600}h {((seconds % 3600) // 60):02d}m"

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    for iteration in range(start_iteration + 1, _C.train.num_iterations + 1):
        data_time = time.perf_counter()
        batch = next(dataloader_iter)
        data_time = time.perf_counter() - data_time

        if sync_cuda_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        timer.tic()
        optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

        with torch.autocast(device_type=device.type, enabled=_C.train.amp):
            if _model_uses_explicit_box_forward(model):
                tokens = tokenizer(batch["text"])
                box_tokens = tokenizer(batch["box_text"])
                output_dict = model(
                    _move_tensor_to_device(batch["image"], device, non_blocking_to_device),
                    _move_tensor_to_device(batch["box_image"], device, non_blocking_to_device),
                    tokens,
                    box_tokens,
                )
            elif _model_uses_box_augmentation(model):
                all_images = torch.cat([batch["image"], batch["box_image"]], dim=0)
                all_texts = batch["text"] + batch["box_text"]
                tokens = tokenizer(all_texts)
                output_dict = model(
                    _move_tensor_to_device(all_images, device, non_blocking_to_device),
                    tokens,
                )
            else:
                tokens = tokenizer(batch["text"])
                output_dict = model(
                    _move_tensor_to_device(batch["image"], device, non_blocking_to_device),
                    tokens,
                    iteration=iteration,
                )

            loss = output_dict["loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        if sync_cuda_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        timer.toc()

        if iteration % _A.log_period == 0:
            now = time.perf_counter()
            log_interval = max(1, iteration - last_log_iteration)
            wall_time_per_iter = (now - last_log_wall_time) / log_interval
            wall_eta = _format_eta(wall_time_per_iter * (_C.train.num_iterations - iteration))
            last_log_wall_time = now
            last_log_iteration = iteration

            timer_stats = (
                f"Iter {timer.iteration} | Time (sec): {data_time:.3f} data, "
                f"{timer.deltas[-1]:.3f} model_dispatch, {wall_time_per_iter:.3f} wall/iter "
                f"| ETA: {wall_eta}"
            )

            log_str = f"{timer_stats} [GPU {dist.gpu_mem_usage()} MB]"
            for key, value in output_dict["logging"].items():
                if not (isinstance(value, list) or isinstance(value, dict)):
                    log_str += f" [{key} {float(value):.3f}]"

            logger.info(log_str)

            if dist.is_main_process():
                log_data = {
                    "iteration": iteration,
                    "scalar": {
                        "lr": scheduler.get_last_lr()[0],
                        "amp_scale": scaler.get_scale(),
                    },
                    "type": "scalar",
                }

                for name, _value in output_dict["logging"].items():
                    prefix = "curv" if name == "curv" else "train"
                    process_logging_value(log_data, prefix, name, _value)

                try:
                    tensorboard_input_queue.put(log_data, timeout=5.0)
                except Exception as e:
                    logger.error(f"Failed to send log data to TensorBoard queue: {e}")
                    logger.warning("Continuing training without TensorBoard logging")

        if iteration % _A.checkpoint_period == 0 and dist.is_main_process():
            checkpoint_manager.step(iteration)

            if run_eval and evaluation_input_queue is not None:
                try:
                    checkpoint_path = output_dir / f"checkpoint_{iteration:08d}.pth"
                    yaml_config_path = output_dir / "config.yaml"
                    evaluation_input_queue.put(
                        (str(checkpoint_path), str(yaml_config_path), iteration)
                    )
                    logger.info(f"Queued evaluation job for iteration {iteration}")
                except Exception as e:
                    logger.error(f"Error in queueing evaluation job: {str(e)}")

    if dist.is_main_process():
        checkpoint_manager.final_step(iteration=_C.train.num_iterations)

    _shutdown_background_workers()


if __name__ == "__main__":
    _A = parser.parse_args()
    if _A.num_gpus == 0:
        main(_A)
    else:
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
        dist_url = f"tcp://{ip_addr}:{_random_port}"

        dist.launch(
            main,
            num_machines=_A.num_machines,
            num_gpus_per_machine=_A.num_gpus,
            machine_rank=_A.machine_rank,
            dist_url=dist_url,
            args=(_A,),
        )
