import torch
import numpy as np
import math
import os
import mdtraj as md
from mdtraj import Trajectory
import warnings
from enum import Enum
from typing import Optional, Any, Callable, Sequence, Union
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data.data import Data


class AtomSelection(Enum):
    PROTEIN = "protein"
    A_CARBON = "c-alpha"
    ALL = "all"


class Molecules(Enum):
    CHIGNOLIN = "CLN025"
    TRP_CAGE = "2JOF"
    BBA = "1FME"
    VILLIN = "2F4K"
    WW_DOMAIN = "GTT"
    NTL9 = "NTL9"
    BBL = "2WAV"
    PROTEIN_B = "PRB"
    HOMEODOMAIN = "UVF"
    PROTEIN_G = "NuG2"
    ALPHA3D = "A3D"
    LAMBDA_REPRESSOR = "lambda"


all_molecules = ["alanine_dipeptide"] + [mol.name.lower() for mol in Molecules]

norm_stds = {
    Molecules.CHIGNOLIN: 3.113133430480957,
    Molecules.TRP_CAGE: 5.08211088180542,
    Molecules.BBA: 6.294918537139893,
    Molecules.VILLIN: 6.082900047302246,
    Molecules.PROTEIN_G: 6.354289531707764,
    "alanine_fold1": 0.9449278712272644,
    "alanine_fold2": 0.944965124130249,
    "alanine_fold3": 0.9452606439590454,
    "alanine_fold4": 0.9454087018966675,
}


def get_dataset(
    mol,
    mean0,
    data_folder=None,
    fold=None,
    traindata_subset=None,
    shuffle_before_splitting=False,
    pdb_folder=None,
):
    """
    Get dataset for a specific molecule.

    Args:
        mol (str): molecule name
        mean0 (bool): whether or not to center at zero
        data_folder: path to folder containing data, run with empty dataset if None
        fold (int in [1,2,3,4]): fold number, only for alanine dipeptide
        traindata_subset (int): subset for training data
        shuffle_before_splitting (bool): whether or not to shuffle the data
        pdb_folder (str): path to folded pdb files, use "datasets/folded_pdbs/" if None (default)

    NB: the relevant file (ala2_cg_2fs_Hmass_2_HBonds.npz) for the alanine dipeptide dataset
    can be downloaded freely from https://ftp.imp.fu-berlin.de/pub/cmb-data/
    """

    if pdb_folder is None:
        pdb_folder = "datasets/folded_pdbs/"
    if mol.lower() == "alanine_dipeptide_fuberlin":
        assert fold is not None and fold in [
            1,
            2,
            3,
            4,
        ], "Please supply a fold in [1,2,3,4]"

        dataset = FUBerlinAlanine2pDataset(data_folder, fold, pdb_folder, mean0=mean0)

        if data_folder is not None:
            idx_range = torch.arange(len(dataset))
            assert (
                not shuffle_before_splitting
            ), f"Shuffling data before split not supported for dataset {mol}."
            allfolds = idx_range.chunk(4)
            testrange = allfolds[fold - 1]
            trainvalrange = torch.cat(allfolds[: fold - 1] + allfolds[fold:])
            # shuffle trainval data
            trainvalrange = trainvalrange[torch.randperm(len(trainvalrange))[:]]
            trainrange = trainvalrange[0:500000]
            valrange = trainvalrange[500000:]
            assert len(trainrange) + len(valrange) == len(trainvalrange)

            if traindata_subset is not None:
                print("should not go here")
                assert (
                    type(traindata_subset) == int
                    and traindata_subset > 0
                    and len(trainrange) >= traindata_subset
                ), "Provide valid number of points for subset"
                trainrange = trainrange[:traindata_subset]

            trainset = dataset.get_subset(trainrange, dataset.topology, train=True)
            valset = dataset.get_subset(valrange, train=False)
            testset = dataset.get_subset(testrange, train=False)
        else:
            trainset = dataset
            valset = dataset
            testset = dataset

    elif "alanine_dipeptide" not in mol.lower():
        # D.E. Shaw fast folding proteins data
        if fold is not None:
            warnings.warn("Fold not implemented for this dataset")
        if traindata_subset is not None:
            warnings.warn(
                "Traindata subset is not implemented for this molecule. Ignoring this argument"
            )

        molecule = Molecules[mol.upper()]
        print(molecule)
        full_simulation_id = "-".join([molecule.value, str(0), "c-alpha"])
        pdb_file = os.path.join(pdb_folder, full_simulation_id + ".pdb")
        topology = md.load_topology(pdb_file)

        if data_folder is None:
            dataset = None
        else:
            dataset = DEShawDataset(
                data_root=data_folder,
                molecule=molecule,
                simulation_id=0,
                atom_selection=AtomSelection.A_CARBON,
                download=True,
                return_bond_graph=False,
                transform=to_angstrom,
                align=False,
            )
        dataset = CGDataset(
            dataset, topology, molecule, mean0=mean0, shuffle=shuffle_before_splitting
        )

        if dataset.dataset is not None:
            valratio, testratio = 0.1, 0.2
            num_val = math.floor(valratio * dataset.__len__())
            num_test = math.floor(testratio * dataset.__len__())
            num_train = dataset.__len__() - num_val - num_test
            idx_range = torch.arange(len(dataset))
            train_idx = idx_range[:num_train]
            val_idx = idx_range[num_train : num_train + num_val]
            test_idx = idx_range[num_train + num_val :]
            trainset = dataset.get_subset(train_idx, topology, train=True)
            valset = dataset.get_subset(val_idx, topology, train=False)
            testset = dataset.get_subset(test_idx, topology, train=False)
        else:
            trainset = dataset
            valset = dataset
            testset = dataset
    else:
        raise Exception(
            f"Wrong dataset mol/dataset name {mol}. Provide valid molecule from {all_molecules}"
        )

    return trainset, valset, testset


def to_angstrom(x):
    """
    Convert from nanometer to angstrom.
    """
    return x * 10.0


class CGDataset(torch.utils.data.TensorDataset):
    """
    Dataset class specific for CG experiments
    Args:
        dataset: atom coordinates
        topology: topology of the molecule
        molecule (str): molecule name
        mean0 (bool): center molecules at zero
        atom_selection (list of ints): list containing atoms to keep
        shuffle: whether or not to shuffle data
    """

    def __init__(
        self,
        dataset,
        topology,
        molecule,
        mean0=True,
        atom_selection=None,
        shuffle=False,
    ):
        self.dataset = dataset
        self.mean0 = mean0
        self.atom_selection = atom_selection
        if dataset is not None:
            self.dataset = self.prepare_dataset(dataset, mean0, atom_selection, shuffle)
        self.topology = topology
        self.molecule = molecule
        self.std = norm_stds[molecule]
        if hasattr(molecule, "name"):
            assert "alanine" not in molecule.name.lower()
            self.num_beads = topology.n_residues
        elif "alanine" in molecule.lower():
            self.num_beads = 5
        else:
            raise NotImplementedError("Invalid molecule name")
        self.bead_onehot = torch.eye(self.num_beads)
        if dataset is None:
            dataset = torch.zeros(1)
        super().__init__(dataset)

    def prepare_dataset(self, dataset, mean0, atom_selection, shuffle):
        """
        Prepare dataset
        """
        data = dataset[:]
        if atom_selection is not None:
            data = data[:, atom_selection, :]
        if mean0:
            data -= data.mean(dim=1, keepdim=True)
        if shuffle:
            data = data.numpy()
            np.random.seed(2342361)
            np.random.shuffle(data)
            data = torch.Tensor(data)
        return data

    def add_attributes(self, topology):
        """
        Add extra attributes to dataset.
        """
        self.topology = topology
        if self.atom_selection is not None:
            self.topology = topology.subset(self.atom_selection)
        self.num_beads = self[:][0].shape[1]
        self.bead_onehot = torch.eye(self.num_beads)
        self.std = norm_stds[self.molecule]

    def get_subset(self, ind_range, topology=None, train=True, forces=False):
        """
        Get subset of entire dataset
        """
        subset = torch.utils.data.Subset(self.dataset, ind_range)
        subset = CGDataset(
            subset, topology, self.molecule, self.mean0, self.atom_selection
        )
        if train:
            assert topology is not None, "Provide topology for train set"
            subset.add_attributes(topology)
        return subset


class FUBerlinAlanine2pDataset(CGDataset):
    """
    Dataset for FU Berlin alanine-dipeptide data. Inherits from CGDataset.
    Args:
        data_root (string): root directory for alanine dipeptide data
        mean0 (bool): whether or not to center at zero

    NB: the relevant file (ala2_cg_2fs_Hmass_2_HBonds.npz) for the alanine dipeptide dataset
    can be downloaded freely from https://ftp.imp.fu-berlin.de/pub/cmb-data/
    """

    def __init__(self, data_root, fold, pdb_folder, mean0=True):
        if data_root is None:
            data_coords = None
        else:
            npz_file = "ala2_cg_2fs_Hmass_2_HBonds.npz"
            local_npz_file = os.path.join(data_root, npz_file)
            data_coords = torch.from_numpy(np.load(local_npz_file)["coords"])

        self.topology = md.load(os.path.join(pdb_folder, "ala2_cg.pdb")).topology

        super().__init__(data_coords, self.topology, f"alanine_fold{fold}", mean0=mean0)


class TemporalSequence:
    def __init__(self, timestep: float):
        """A sequence with a timestep attribute indicating the time between consecutive elements.

        Args:
            timestep (float): The time resolution of the sequence in picoseconds.
        """
        self.timestep = timestep


class MDTrajectory(TemporalSequence, Dataset):
    def __init__(
        self,
        traj: Trajectory,
        extra_features: Sequence = None,
        transform: Optional[Callable[[Any], Any]] = None,
        return_bond_graph: bool = False,
        timestep: Optional[float] = None,
        align: bool = False,
    ):
        """Dataset object for a Molecular Dinamics Trajectory object from the mdtraj module

        Args:
            traj (Trajectory): the trajectory to be transformed into a dataset
            extra_features (Sequence): A sequence of (labeled) features to return (other than positions).
            transform (Optional[Callable[[Any] ,Any]], optional): A function to be applied to the atom coordinates (after indexing). Defaults to None.
            return_bond_graph (bool, optional): flag to specify if the data-items are graphs or coordinates only. Defaults to False.
            timestep (float, optional): the time in between consetutive simulation frames in picoseconds. Defaults to None.
            align (bool): flat to align the Trajectories. Defaults to False.
        """

        # Align the trajectory if required
        if align:
            traj = traj.superpose(traj, 0)

        # Save the original trajectory
        self.traj = traj
        self.return_bond_graph = return_bond_graph
        if not (extra_features is None):
            assert len(extra_features) == len(
                traj.xyz
            ), "The extra features must have the same lenght as the trajectory"
        self.extra_features = extra_features
        self.transform = transform

        # Make a lookup dictionary for the atoms in the molecule
        self.atomsdict = {atom: i for i, atom in enumerate(traj.topology.atoms)}

        # Compute an edge index based on the bonds if required
        if return_bond_graph:
            self.edge_index = torch.LongTensor(
                [
                    [self.atomsdict[edge[0]], self.atomsdict[edge[1]]]
                    for edge in traj.topology.bonds
                ]
            ).T

        if timestep is None:
            timestep = traj.timestep

        super(MDTrajectory, self).__init__(timestep=timestep)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Data]:
        x = torch.FloatTensor(self.traj.xyz[idx])
        atom_ids = [a.element.atomic_number - 1 for a in self.traj.topology.atoms]

        if self.return_bond_graph:
            # wrap into a torch_geometric graph
            x = Data(
                pos=x, atom_labels=torch.tensor(atom_ids), edge_index=self.edge_index
            )

        if not (self.transform is None):
            x = self.transform(x)

        # Add the extra features
        if not (self.extra_features is None):
            extra_features = self.extra_features[idx]

            # if the extra features are a dictionary, make sure the is no key 'x'
            if isinstance(extra_features, dict):
                assert not (
                    x in extra_features
                ), "The extra features can't specify a key named 'x'"

            # In case a graph is returned
            if self.return_bond_graph:
                if not isinstance(extra_features, dict):
                    extra_features = {"y": extra_features}
                # Add the attributes to the data object
                for k, v in extra_features.items():
                    setattr(x, k, v)
            else:
                if isinstance(extra_features, dict):
                    # add x to the features
                    extra_features["x"] = x
                    x = extra_features
                else:
                    x = (x, extra_features)
        return x

    def __len__(self):
        return len(self.traj.xyz)


class DEShawDataset(MDTrajectory):
    def __init__(
        self,
        data_root: str,
        molecule: Molecules,
        simulation_id: int,
        atom_selection: AtomSelection = AtomSelection.PROTEIN,
        transform: Optional[Callable[[Any], Any]] = None,
        return_bond_graph: bool = False,
        align: bool = False,
    ):
        self.data_root = data_root
        self.simulation_id = simulation_id
        self.atom_selection = atom_selection

        full_simulation_id = "-".join(
            [molecule.value, str(simulation_id), atom_selection.value]
        )

        simulation_path = os.path.join(
            molecule.value,
            f"simulation_{simulation_id}",
            atom_selection.value,
            full_simulation_id,
        )
        full_simulation_id = simulation_path.split("/")[-1]

        time_file = os.path.join(simulation_path, full_simulation_id + "_times.csv")
        pdb_file = os.path.join(simulation_path, full_simulation_id + ".pdb")

        time_data = pd.read_csv(
            os.path.join(self.data_root, time_file), names=["time", "file"]
        )

        local_trajectory_files = [
            os.path.join(self.data_root, simulation_path, trajectory_part_file)
            for trajectory_part_file in time_data["file"].values
        ]
        local_pdb_file = os.path.join(self.data_root, pdb_file)

        # Load the trajectory using mdtraj
        traj = md.load(local_trajectory_files, top=local_pdb_file)

        super(DEShawDataset, self).__init__(
            traj=traj,
            transform=transform,
            return_bond_graph=return_bond_graph,
            timestep=time_data["time"].values[0],
            align=align,
        )
