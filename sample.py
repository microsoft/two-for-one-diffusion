import argparse
import pickle
from os.path import join
from pathlib import Path
import torch
from models import get_model
from models.ddpm import GaussianDiffusion
from ema_pytorch import EMA
from datasets.dataset_utils_empty import get_dataset
from evaluate.evaluators import sample_from_model
from dynamics.langevin import LangevinDiffusion
from utils import SamplerWrapper
from dynamics.langevin import temp_dict
import mdtraj as md
from torch.utils.tensorboard import SummaryWriter
import time

parser = argparse.ArgumentParser(description="coarse-graining-evaluator")
parser.add_argument(
    "--model_path",
    type=str,
    help="root directory where models and args are stored",
    required=True,
)
parser.add_argument(
    "--model_checkpoint", type=str, default="best", help="best, last, 1, 2, 3, ..."
)
parser.add_argument(
    "--gen_mode",
    type=str,
    default="iid",
    help="generative mode, either iid or langevin",
)
parser.add_argument(
    "--append_exp_name",
    type=str,
    default=None,
    help="append this text to the results/main_eval_output folder name, append only gen_mode if None (default)",
)
parser.add_argument(
    "--data_folder",
    type=str,
    default=None,
    help="directory root where data is stored, if None (default) work with empty datasets and saved reference from saved_histograms",
)


# i.i.d. generation arguments
parser.add_argument(
    "--num_samples_eval",
    type=int,
    default=1000,
    help="number of samples for i.i.d. generation",
)
parser.add_argument(
    "--batch_size_gen", type=int, default=256, help="batch size for evaluation"
)


# Langevin simulation arguments
parser.add_argument("--masses", type=eval, default=None, help="Units in g/mol")
parser.add_argument(
    "--friction",
    type=float,
    default=1,
    help="No units yet. Ideally units should be in ps^-1, usually 1",
)
parser.add_argument(
    "--parallel_sim", type=int, default=100, help="Number of parallel simulations"
)
parser.add_argument(
    "--n_timesteps", type=int, default=10000, help="number of timesteps"
)
parser.add_argument(
    "--save_interval", type=int, default=250, help="save interval (in timesteps)"
)
parser.add_argument(
    "--noise_level",
    type=int,
    default=20,
    help="diffusion model noise level for extracting force fields",
)
parser.add_argument(
    "--dt",
    type=float,
    default=None,
    help="Ideally 1~2fs (units in ps), if None it will be computed automatically according to the diffusion model parameters",
)
parser.add_argument(
    "--temp_data", type=float, default=None, help="temperature in Kelvin."
)
parser.add_argument(
    "--temp_sim", type=float, default=None, help="temperature in Kelvin"
)
parser.add_argument("--kb", type=str, default="consistent", help="consistent, kcal")


samp_args = parser.parse_args()


def main(samp_args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load args from training
    with open(join(samp_args.model_path, "args.pickle"), "rb") as f:
        args = pickle.load(f)

    if samp_args.temp_data is None:
        samp_args.temp_data = temp_dict[args.mol.upper()]
    if samp_args.temp_sim is None:
        samp_args.temp_sim = temp_dict[args.mol.upper()]
    else:
        samp_args.temp_sim = samp_args.temp_sim

    basic_append = f"_{samp_args.gen_mode}"
    samp_args.append_exp_name = (
        basic_append
        if samp_args.append_exp_name is None
        else f"{basic_append}_{samp_args.append_exp_name}"
    )

    eval_folder = Path(
        join(samp_args.model_path, "main_eval_output" + samp_args.append_exp_name)
    )

    args.data_folder = samp_args.data_folder
    eval_folder.mkdir(exist_ok=True, parents=False)
    writer = SummaryWriter(str(eval_folder))

    # Load dataset from args
    trainset, _, _ = get_dataset(
        args.mol,
        args.mean0,
        args.data_folder,
        args.fold,
        shuffle_before_splitting=args.shuffle_data_before_splitting,
    )

    norm_factor = trainset.std if args.scale_data else 1.0

    # Init model from args
    model_nn = get_model(args, trainset, device)
    print(model_nn)

    # Init DDPM from args
    DDPM_model = GaussianDiffusion(
        model=model_nn,
        features=trainset.bead_onehot,
        num_atoms=trainset.num_beads,
        timesteps=args.diffusion_steps,
        norm_factor=norm_factor,
        loss_weights=args.loss_weights,
    ).to(device)
    model = EMA(DDPM_model)

    # Load weights into model
    if torch.cuda.is_available():
        data_dict = torch.load(
            samp_args.model_path + f"/model-{samp_args.model_checkpoint}.pt"
        )
    else:
        data_dict = torch.load(
            samp_args.model_path + f"/model-{samp_args.model_checkpoint}.pt",
            map_location=torch.device("cpu"),
        )

    model.load_state_dict(data_dict["ema"])

    generate_samples(model, trainset, samp_args.noise_level, args, device, eval_folder)

    writer.flush()
    writer.close()
    time.sleep(2)


def generate_samples(model, trainset, noise_level, args, device, eval_folder):
    # Generate iid samples
    if samp_args.gen_mode == "iid":
        sampler = SamplerWrapper(model.ema_model).to(device).eval()
        if torch.cuda.device_count() > 1 and device == "cuda":
            sampler = torch.nn.DataParallel(sampler).to(device)
            parallel_batches = torch.cuda.device_count()
        else:
            parallel_batches = 1
        sampled_mol = sample_from_model(
            sampler,
            samp_args.num_samples_eval // parallel_batches,
            samp_args.batch_size_gen // parallel_batches,
            verbose=True,
        )
    # Generate Langevin samples from simulation
    elif samp_args.gen_mode == "langevin":
        print(
            f"Total number of samples to save using Langevin Dynamics: {int(samp_args.parallel_sim * samp_args.n_timesteps / samp_args.save_interval)}"
        )

        # NOTE: instead of drawing initial samples from the training set (commented below),
        # draw samples for initial states (assume dataset not available).

        # dl = data.DataLoader(trainset, batch_size=eval_args.parallel_sim, shuffle=True)
        # init_mol = next(iter(dl))[0]

        sampler = SamplerWrapper(model.ema_model).to(device).eval()
        if torch.cuda.device_count() > 1 and device == "cuda":
            sampler = torch.nn.DataParallel(sampler).to(device)
            parallel_batches = torch.cuda.device_count()
        else:
            parallel_batches = 1
        init_mol = sample_from_model(
            sampler,
            samp_args.parallel_sim // parallel_batches,
            samp_args.batch_size_gen // parallel_batches,
            verbose=True,
        )

        masses = samp_args.masses
        if masses is None:
            if "alanine" in args.mol:
                masses = [12.8] * trainset.num_beads
            else:
                masses = [12.0] * trainset.num_beads

        langevin_sampler = LangevinDiffusion(
            model.ema_model,
            init_mol,
            samp_args.n_timesteps,
            save_interval=samp_args.save_interval,
            t=noise_level,
            diffusion_steps=args.diffusion_steps,
            temp_data=samp_args.temp_data,
            temp_sim=samp_args.temp_sim,
            dt=samp_args.dt,
            masses=masses,
            friction=samp_args.friction,
            kb=samp_args.kb,
        )
        sampled_mol = langevin_sampler.sample()
    else:
        raise Exception("Wrong argument 'gen_mode'")

    # Save generated samples
    torch.save(sampled_mol, str(str(eval_folder) + f"/sample-{samp_args.gen_mode}.pt"))
    # Save subset as pdb
    all_mol_traj = md.Trajectory(
        sampled_mol[0:1000].numpy() / 10, topology=trainset.topology
    )
    all_mol_traj.save_pdb(str(str(eval_folder) + f"/sample-{samp_args.gen_mode}.pdb"))

    return sampled_mol


if __name__ == "__main__":
    main(samp_args)
