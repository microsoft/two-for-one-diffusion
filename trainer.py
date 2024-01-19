import torch
import datetime as dt
from pathlib import Path
from tqdm.auto import tqdm
from cmath import inf
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import cpu_count
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from ema_pytorch import EMA
import pickle
from evaluate.evaluators import Evaluator, sample_from_model
from dynamics.langevin import temp_dict, temp_dict_pt, LangevinDiffusion
import utils
import time

from utils import (
    cycle,
    random_rotation,
)


class Trainer(object):
    """
    Trainer for diffusion model for coarse-grained MD.
    Check --help of main_train for argument details.
    """

    def __init__(
        self,
        diffusion_model,
        dataset,  # tuple: (train_data, val_data, test_data)
        mol_name,
        args,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=1,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        save_and_sample_every=1000,
        results_folder="./results",
        num_saved_samples=10,
        topology=None,
        data_aug=True,
        tb_folder="./runs",
        experiment_name="",
        weight_decay=0,
        log_tensorboard_interval: int = 1,
        num_samples_final_eval=100,
        min_lr_cosine_anneal=None,
        eval_langevin=False,
        langevin_timesteps=1000000,
        langevin_stepsize=2e-3,  # picoseconds
        langevin_t_diffs=[12],
        pick_checkpoint="best",  # last, best
        start_from_last_saved=False,
        iterations_on_val=1,
        t_diff_interval=None,
        save_all_checkpoints=False,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = diffusion_model
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(
            self.device
        )
        self.sampler_ema = utils.SamplerWrapper(self.ema.ema_model).to(self.device)

        if torch.cuda.device_count() > 1 and self.device == "cuda":
            self.model_dp = torch.nn.DataParallel(self.model).to(self.device)
            self.model_ema_dp = torch.nn.DataParallel(self.ema.ema_model).to(
                self.device
            )
            self.sampler_ema_dp = torch.nn.DataParallel(self.sampler_ema).to(
                self.device
            )
            self.parallel_batches = torch.cuda.device_count()
        else:
            self.model_dp = self.model.to(self.device)
            self.model_ema_dp = self.ema.ema_model.to(self.device)
            self.sampler_ema_dp = self.sampler_ema.to(self.device)
            self.parallel_batches = 1

        self.args = args

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.num_atoms = diffusion_model.num_atoms
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.log_tensorboard_interval = log_tensorboard_interval
        self.langevin_timesteps = langevin_timesteps
        self.langevin_stepsize = langevin_stepsize
        self.langevin_t_diffs = langevin_t_diffs
        self.mol_name = mol_name
        self.train_data, self.val_data, self.test_data = dataset
        self.pick_checkpoint = pick_checkpoint
        self.t_diff_interval = t_diff_interval
        self.save_all_checkpoints = save_all_checkpoints

        num_workers = min(cpu_count(), 8)

        self.dl_train = cycle(
            data.DataLoader(
                self.train_data,
                batch_size=train_batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=num_workers,
                drop_last=True,
            )
        )
        self.dl_val = data.DataLoader(
            self.val_data,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True,
        )
        self.val_iters = iterations_on_val * len(self.dl_val)
        self.dl_val = cycle(self.dl_val)

        self.opt = AdamW(
            self.model.parameters(), lr=train_lr, weight_decay=weight_decay
        )

        if min_lr_cosine_anneal is not None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, train_num_steps, eta_min=min_lr_cosine_anneal
            )
        self.data_aug = data_aug
        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.num_saved_samples = num_saved_samples
        self.num_samples_final_eval = num_samples_final_eval
        self.topology = topology

        # Tensorboard writer
        tzinfo = dt.timezone(dt.timedelta(hours=2))  # timezone UTC+2
        now = dt.datetime.now(tzinfo)
        if len(experiment_name) > 0:
            experiment_name += "_"
        experiment_name = experiment_name  # + now.strftime("%Y-%m-%d_%X_%Z")
        self.writer = SummaryWriter(tb_folder + "/" + experiment_name + "_trn")
        # self.writer_val = SummaryWriter(tb_folder + "/" + experiment_name + "_val")

        # Results folder from the new folder name
        self.results_folder = Path(results_folder + "/" + experiment_name)

        self.evaluator_val = Evaluator(
            self.val_data,
            self.train_data.topology,
            mol_name=mol_name,
            eval_folder=str(self.results_folder),
            data_folder=args.data_folder,
        )
        self.evaluator_test = Evaluator(
            self.test_data,
            self.train_data.topology,
            mol_name=mol_name,
            eval_folder=str(self.results_folder),
            data_folder=args.data_folder,
        )
        self.eval_langevin = eval_langevin
        self.best_val_loss = inf
        if start_from_last_saved:
            try:
                self.load()
                print("Settings loaded from last checkpoint")
            except:
                print("Not last checkpoint available to load.")

    def save(self, milestone: dict, save_best: bool = False):
        """
        Save model checkpoint
        """
        data_dict = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.scaler.state_dict(),
            "opt": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        # Save the model
        if self.save_all_checkpoints:
            torch.save(data_dict, str(self.results_folder / f"model-{milestone}.pt"))
        torch.save(data_dict, str(self.results_folder / "model-last.pt"))

        # Save the best model if that is the case
        if save_best:
            torch.save(data_dict, str(self.results_folder / "model-best.pt"))

        # Save the arguments
        with open(str(self.results_folder / "args.pickle"), "wb") as f:
            pickle.dump(self.args, f)

    def load(self, milestone="last"):
        """
        Load model checkpoint
        """
        data_dict = torch.load(str(self.results_folder / f"model-{milestone}.pt"))

        self.step = data_dict["step"]
        self.best_val_loss = data_dict["best_val_loss"]
        self.model.load_state_dict(data_dict["model"])
        self.ema.load_state_dict(data_dict["ema"])
        self.scaler.load_state_dict(data_dict["scaler"])
        self.opt.load_state_dict(data_dict["opt"])
        self.scheduler.load_state_dict(data_dict["scheduler"])

    def eval_loss(self, dl, val_iters, partition_name="val", t_diff_range=None):
        print(f"val iters {val_iters}")
        self.model_ema_dp.eval()
        with torch.no_grad():
            iter_num = 0
            loss = 0
            while iter_num < val_iters:
                val_data = next(dl)[0].to(self.device)
                loss += self.model_ema_dp(val_data, t_diff_range=t_diff_range).mean()
                iter_num += 1
            loss /= val_iters
            self.writer.add_scalar(f"Loss {partition_name}", loss.item(), self.step)
            print(f"Loss {partition_name} \t {loss.item()}")
        return loss

    def train(self):
        """
        Train model
        """
        self.model.train()
        self.model_dp.train()
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            early_stopping_counter = 0
            while self.step < self.train_num_steps:
                for i in range(self.gradient_accumulate_every):
                    input = next(self.dl_train)[0].to(self.device)
                    if self.data_aug:
                        input = random_rotation(input)

                    with autocast(enabled=self.amp):
                        loss = self.model_dp(
                            input, t_diff_range=self.t_diff_interval
                        ).mean()
                        self.scaler.scale(
                            loss / self.gradient_accumulate_every
                        ).backward()

                    pbar.set_description(f"loss: {loss.item():.4f}")

                    if self.step % self.log_tensorboard_interval == 0:
                        self.writer.add_scalar("Loss", loss.item(), self.step)

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                self.ema.update()
                self.step += 1
                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.model_ema_dp.eval()
                    milestone = self.step // self.save_and_sample_every
                    val_loss_ff = self.eval_loss(
                        self.dl_val, self.val_iters, partition_name="val"
                    )

                    # Evaluate i.i.d.
                    self.results_folder.mkdir(exist_ok=True, parents=True)
                    sampled_mol = sample_from_model(
                        self.sampler_ema_dp,
                        self.num_saved_samples // self.parallel_batches,
                        self.batch_size // self.parallel_batches,
                    )
                    results_dict = self.evaluator_val.eval(
                        sampled_mol, milestone=str(milestone) + "_iid", save_plots=True
                    )
                    # Write metrics to Tensorboard
                    for key in results_dict:
                        self.writer.add_scalar(key, results_dict[key], self.step)

                    bool_new_best = val_loss_ff.item() < self.best_val_loss
                    self.best_val_loss = (
                        val_loss_ff.item() if bool_new_best else self.best_val_loss
                    )
                    self.save(milestone, save_best=bool_new_best)
                    self.model.train()
                    self.model_dp.train()

                    if not bool_new_best:
                        early_stopping_counter += 1
                    else:
                        early_stopping_counter = 0
                    if early_stopping_counter > 9:
                        break

                pbar.update(1)
                if hasattr(self, "scheduler"):
                    self.scheduler.step()

        print("\nFinal and larger evaluation")
        # Evaluate and save metrics again once finished training

        if self.pick_checkpoint == "best":
            self.load(milestone="best")

        sampled_mol = sample_from_model(
            self.sampler_ema_dp,
            self.num_samples_final_eval,
            self.batch_size // self.parallel_batches,
        )
        if "alanine" not in self.mol_name:
            utils.save_samples(
                sampled_mol,
                str(self.results_folder),
                self.topology,
                milestone="final_iid",
            )

        results_val_dict = self.evaluator_val.eval(
            sampled_mol, milestone="final_iid_val", save_plots=True
        )
        results_test_dict = self.evaluator_test.eval(
            sampled_mol, milestone="final_iid_test", save_plots=False
        )

        # Write metrics to Tensorboard
        for key in results_val_dict:
            self.writer.add_scalar(key + "_FINAL_iid_val", results_val_dict[key])
        # Write metrics to Tensorboard
        for key in results_test_dict:
            self.writer.add_scalar(key + "_FINAL_iid_test", results_test_dict[key])

        if self.eval_langevin:
            for langevin_t_diff in self.langevin_t_diffs:
                temp_data = temp_dict[self.mol_name.upper()]
                temp_pt = temp_dict_pt[self.mol_name.upper()]
                temps_sim = [temp_data]
                for temp_sim in temps_sim:
                    # 100 to match the procedure in
                    dl = data.DataLoader(self.train_data, batch_size=100, shuffle=True)
                    init_mol = next(iter(dl))[0]
                    mass = 12.8 if "alanine".upper() in self.mol_name.upper() else 12
                    save_interval = (
                        250 if "alanine".upper() in self.mol_name.upper() else 200
                    )
                    self.ema.ema_model.eval()
                    langevin_sampler = LangevinDiffusion(
                        self.ema.ema_model,
                        init_mol,
                        self.langevin_timesteps,
                        save_interval=save_interval,
                        t=langevin_t_diff,
                        diffusion_steps=self.args.diffusion_steps,
                        temp_data=temp_data,
                        temp_sim=temp_sim,
                        dt=self.langevin_stepsize,
                        masses=[mass] * self.train_data.num_beads,
                    )
                    sampled_mol = langevin_sampler.sample()
                    if "alanine" not in self.mol_name:
                        utils.save_samples(
                            sampled_mol,
                            str(self.results_folder),
                            self.topology,
                            milestone=f"final_langevin_tdiff{langevin_t_diff}",
                        )
                    results_val_dict = self.evaluator_val.eval(
                        sampled_mol,
                        milestone=f"final_langevin_tdiff{langevin_t_diff}_val",
                        save_plots=True,
                    )
                    results_test_dict = self.evaluator_test.eval(
                        sampled_mol,
                        milestone=f"final_langevin_tdiff{langevin_t_diff}_test",
                        save_plots=False,
                    )
                    # Write metrics to Tensorboard
                    for key in results_val_dict:
                        self.writer.add_scalar(
                            key + f"_FINAL_langevin_t{langevin_t_diff}_val",
                            results_val_dict[key],
                        )
                        self.writer.add_scalar(
                            key + f"_FINAL_langevin_t{langevin_t_diff}_test",
                            results_test_dict[key],
                        )
        self.writer.flush()
        self.writer.close()
        print("Training complete")

        # Wait one second to ensure amulet registers last tensorboard results.
        time.sleep(2)
