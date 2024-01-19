from torch import nn
import torch
import numpy as np
from dynamics.langevin_cgnet import Langevin

KBOLTZMANN = 1.38064852e-23
AVOGADRO = 6.022140857e23
JPERKCAL = 4184
KB = 0.83144626181  # This is the Boltzmann constant conversed from J/K (Kg, m^2 / s^2 / K) to -> g/mol, angstroms, ps and K.

temp_dict = {
    "alanine_dipeptide_fuberlin".upper(): 300,
    "alanine_dipeptide_mdshare".upper(): 300,
    "CHIGNOLIN": 340,
    "TRP_CAGE": 290,
    "BBA": 325,
    "VILLIN": 360,
    "WW_DOMAIN": 360,
    "NTL9": 355,
    "BBL": 298,
    "PROTEIN_B": 340,
    "HOMEODOMAIN": 360,
    "PROTEIN_G": 350,
    "ALPHA3D": 370,
    "LAMBDA_REPRESSOR": 350,
}

temp_dict_pt = {
    "alanine_dipeptide_fuberlin".upper(): 450,
    "alanine_dipeptide_mdshare".upper(): 450,
    "CHIGNOLIN": 500,
    "TRP_CAGE": 500,
    "BBA": 500,
    "VILLIN": 500,
    "WW_DOMAIN": 500,
    "NTL9": 500,
    "BBL": 500,
    "PROTEIN_B": 500,
    "HOMEODOMAIN": 500,
    "PROTEIN_G": 500,
    "ALPHA3D": 500,
    "LAMBDA_REPRESSOR": 500,
}


class ForcesWrapper(nn.Module):
    """
    This class takes as input a diffusion model and converts it into a forces field
    """

    def __init__(
        self,
        model_diff,
        t=10,
        diffusion_steps=1000,
        kbt_inv=1.0,
    ):
        """
        Args:
            model_diff: Diffusion model
            t: The diffusion timestep
            diffusion_steps: the number of diffusion steps
            kbt_inv: the inverse of the boltzmann constant times the temperature
        """
        super(ForcesWrapper, self).__init__()
        self.model_gnn = model_diff.model.eval()
        self.t = torch.Tensor([t]).to(model_diff.device)
        self.sqrt_one_minus_alphas_cumprod = model_diff.sqrt_one_minus_alphas_cumprod[t]
        self.sqrt_alphas_cumprod = model_diff.sqrt_alphas_cumprod
        self.t_norm = self.t / float(diffusion_steps)
        self.kbt_inv = kbt_inv
        self.one_hot = model_diff.h
        self.norm = None

    def forward(self, x_old, embeddings=None):
        t_norm = self.t_norm.reshape(-1, 1, 1).repeat(x_old.shape[0], 1, 1)
        self.t_tensor = self.t.repeat(x_old.shape[0]).long()
        forces = (
            -self.model_gnn(
                x_old,
                self.one_hot,
                t_norm,
                alphas=self.sqrt_alphas_cumprod[self.t_tensor].pow(2),
            )
            / self.kbt_inv
            / self.sqrt_one_minus_alphas_cumprod
        )

        if self.norm is None:
            self.norm = torch.mean(torch.norm(forces.cpu(), dim=2))
            print(f"Forces (norm) {self.norm}")
        return torch.zeros(x_old.shape[0]), forces


class LangevinDiffusion:
    """
    This class simulates the Langevin Dynamics from a diffusion model
    """

    def __init__(
        self,
        model_diff,
        init_mol,
        n_timesteps=1000000,
        save_interval=250,
        t=15,
        diffusion_steps=1000,
        temp_data=300,
        temp_sim=300,
        dt=2e-3,
        masses=[12.8] * 5,
        friction=1,
        kb="consistent",
        exchange_interval=5000,
    ):
        """
        Args:
            model_diff: Diffusion model
            init_mol: pytorch tensor with the init samples
            n_timesteps: the number of steps in the simulation
            save_interval: save a step every 'save_interval' steps
            t: The diffusion timestep
            diffusion_steps: the number of diffusion steps
            temp: temperature of the simulation, usually given by the training data.
            dt: time resolution through the simulation
            masses: list with the mass of each element in the graph
            friction: friction constnat in the Langevin Simulation
            kb: what boltzmann constant to use (consistent, kcal)
        """
        print(f"norm factor:{model_diff.norm_factor}")
        self.norm_factor = (
            model_diff.norm_factor
        )  # / model_diff.sqrt_alphas_cumprod[t].item()
        # print(self.norm_factor)
        init_sample = init_mol / self.norm_factor
        self.device = model_diff.device
        self.one_minus_alphas_cumprod = 1 - model_diff.alphas_cumprod[t].item()

        if kb == "consistent":
            self.kb_inv = 1 / KB * self.norm_factor**2
        elif kb == "kcal":
            self.kb_inv = (
                JPERKCAL / KBOLTZMANN / AVOGADRO * (self.norm_factor**2) / 100
            )
        else:
            raise Exception("Wrong kb value")

        self.model_forces = ForcesWrapper(
            model_diff,
            t,
            diffusion_steps,
            kbt_inv=self.kb_inv / temp_data,
        )

        if friction is None:
            friction_aux = 1
            diffusion_constant = 1 / masses[0]
        else:
            friction_aux = friction
            diffusion_constant = 1
        if dt is None:
            dt = (
                self.one_minus_alphas_cumprod
                * friction_aux
                * masses[0]
                * self.kb_inv
                / temp_data
            )

        self.sim = Langevin(
            self.model_forces,
            init_sample,
            length=n_timesteps,
            save_interval=save_interval,
            beta=self.kb_inv / temp_sim,
            save_potential=False,
            device=self.device,
            log_interval=save_interval,
            log_type="print",
            diffusion=diffusion_constant,
            masses=masses,
            friction=friction,
            dt=dt,
        )

        print(f"Diffusion model Beta : {model_diff.betas[t]}")
        print(
            f"Diffusion model sqrt_alphas_cumprod {model_diff.sqrt_alphas_cumprod[t]}"
        )
        print(
            f"Diffusion model sqrt_one_minus_alphas_cumprod {model_diff.sqrt_one_minus_alphas_cumprod[t]}"
        )
        print(
            f"Diffusion model one_minus_alphas_cumprod {self.one_minus_alphas_cumprod}"
        )
        if friction is None:
            friction = 1
        print(
            f"dt*kb*T/M/gamma: {dt * temp_data / self.kb_inv / masses[0] / friction} (should be on a similar scale as one_minus_alphas_cumprod)"
        )

        print(f"dt: {dt: .8f} (ps)")
        print(f"KbT: {temp_data/self.kb_inv: .4f}")

    def sample(self):
        # Simulate
        traj = self.sim.simulate()
        # Cast all samples into the batch size
        traj = torch.Tensor(traj)
        traj = traj.reshape(-1, traj.size(2), traj.size(3))
        traj = traj * self.norm_factor
        return traj
