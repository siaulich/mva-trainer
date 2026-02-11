"""Evaluator for comparing event reconstruction methods."""

import numpy as np
from .physics_calculations import (
    TopReconstructor,
    ResolutionCalculator,
    lorentz_vector_from_PtEtaPhiE_array,
    c_hel,
    c_han,
)
from core.utils import (
    compute_pt_from_lorentz_vector_array,
    lorentz_vector_from_neutrino_momenta_array,
    compute_mass_from_lorentz_vector_array,
    project_vectors_onto_axis,
    angle_vectors,
    cos_angle_between_vectors,
)

# Function aliases
make_4vect = lorentz_vector_from_PtEtaPhiE_array
make_nu_4vect = lorentz_vector_from_neutrino_momenta_array


reconstruction_variable_configs = {
    "top_mass": {
        "compute_func": lambda l, j, n: TopReconstructor.compute_top_masses(
            (l[:, 0, :4] + j[:, 0, :4] + n[:, 0, :]) / 1e3
        ),
        "extract_func": lambda X: TopReconstructor.compute_top_masses(
            make_4vect(X["top_truth"][:, 0, :4]) / 1e3
        ),
        "label": r"$m(t)$ [GeV]",
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $m(t)$ Resolution",
            "ylabel_deviation": r"Mean Relative $m(t)$ Deviation",
        },
    },
    "ttbar_mass": {
        "compute_func": lambda l, j, n: TopReconstructor.compute_ttbar_mass(
            (l[:, 0, :4]) + (j[:, 0, :4]) + (n[:, 0, :]),
            (l[:, 1, :4]) + (j[:, 1, :4]) + (n[:, 1, :]),
        )
        / 1e3,
        "extract_func": lambda X: TopReconstructor.compute_ttbar_mass(
            make_4vect(X["top_truth"][:, 0, :4]),
            make_4vect(X["top_truth"][:, 1, :4]),
        )
        / 1e3,
        "label": r"$m(t\overline{t})$ [GeV]",
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $m(t\overline{t})$ Resolution",
            "ylabel_deviation": r"Mean Relative $m(t\overline{t})$ Deviation",
        },
    },
    "c_han": {
        "compute_func": lambda l, j, n: c_han(
            *TopReconstructor.compute_top_lorentz_vectors(l, j, n),
            (l[:, 0, :4]),
            (l[:, 1, :4]),
        ),
        "extract_func": lambda X: c_han(
            make_4vect(X["top_truth"][:, 0, :4]),
            make_4vect(X["top_truth"][:, 1, :4]),
            make_4vect(X["lepton_truth"][:, 0, :4]),
            make_4vect(X["lepton_truth"][:, 1, :4]),
        ),
        "label": r"$c_{\text{han}}$",
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$c_{\text{han}}$ Resolution",
            "ylabel_deviation": r"Mean $c_{\text{han}}$ Deviation",
        },
    },
    "c_hel": {
        "compute_func": lambda l, j, n: c_hel(
            *TopReconstructor.compute_top_lorentz_vectors(l, j, n),
            (l[:, 0, :4]),
            (l[:, 1, :4]),
        ),
        "extract_func": lambda X: c_hel(
            make_4vect(X["top_truth"][:, 0, :4]),
            make_4vect(X["top_truth"][:, 1, :4]),
            make_4vect(X["lepton_truth"][:, 0, :4]),
            make_4vect(X["lepton_truth"][:, 1, :4]),
        ),
        "label": r"$c_{\text{hel}}$",
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$c_{\text{hel}}$ Resolution",
            "ylabel_deviation": r"Mean $c_{\text{hel}}$ Deviation",
        },
    },
    "nu_mag": {
        "compute_func": lambda l, j, n: (np.linalg.norm(n[:, 0, :], axis=-1) / 1e3,),
        "extract_func": lambda X: (
            np.linalg.norm(X["regression"][:, 0, :], axis=-1) / 1e3,
        ),
        "label": r"$|\vec{p}(\nu)|$ [GeV]",
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $|\vec{p}(\nu)|$ Resolution",
            "ylabel_deviation": r"Mean Relative $|\vec{p}(\nu)|$ Deviation",
        },
    },
    "nubar_mag": {
        "compute_func": lambda l, j, n: (np.linalg.norm(n[:, 1, :], axis=-1) / 1e3,),
        "extract_func": lambda X: (
            np.linalg.norm(X["regression"][:, 1, :], axis=-1) / 1e3,
        ),
        "label": r"$|\vec{p}(\overline{\nu})|$ [GeV]",
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $|\vec{p}(\overline{\nu})|$ Resolution",
            "ylabel_deviation": r"Mean Relative $|\vec{p}(\overline{\nu})|$ Deviation",
        },
    },
    "top_pt": {
        "compute_func": lambda l, j, n: (
            compute_pt_from_lorentz_vector_array(
                TopReconstructor.compute_top_lorentz_vectors(l, j, n)[0]
            )
            / 1e3
        ),
        "extract_func": lambda X: (
            compute_pt_from_lorentz_vector_array((X["top_truth"][:, 0, :4])) / 1e3
        ),
        "label": r"$p_{T}(t)$ [GeV]",
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $p_{T}(t)$ Resolution",
            "ylabel_deviation": r"Mean Relative $p_{T}(t)$ Deviation",
        },
    },
    "nu_px": {
        "compute_func": lambda l, j, n: (n[:, 0, 0] / 1e3),
        "extract_func": lambda X: (X["regression"][:, 0, 0] / 1e3),
        "label": r"$p_{x}(\nu)$ [GeV]",
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{x}(\nu)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{x}(\nu)$ Deviation [GeV]",
        },
    },
    "nu_py": {
        "compute_func": lambda l, j, n: (n[:, 0, 1] / 1e3),
        "extract_func": lambda X: (X["regression"][:, 0, 1] / 1e3),
        "label": r"$p_{y}(\nu)$ [GeV]",
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{y}(\nu)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{y}(\nu)$ Deviation [GeV]",
        },
    },
    "nu_pz": {
        "compute_func": lambda l, j, n: (n[:, 0, 2] / 1e3),
        "extract_func": lambda X: (X["regression"][:, 0, 2] / 1e3),
        "label": r"$p_{z}(\nu)$ [GeV]",
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{z}(\nu)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{z}(\nu)$ Deviation [GeV]",
        },
    },
    "nubar_px": {
        "compute_func": lambda l, j, n: (n[:, 1, 0] / 1e3),
        "extract_func": lambda X: (X["regression"][:, 1, 0] / 1e3),
        "label": r"$p_{x}(\overline{\nu})$ [GeV]",
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{x}(\overline{\nu})$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{x}(\overline{\nu})$ Deviation [GeV]",
        },
    },
    "nubar_py": {
        "compute_func": lambda l, j, n: (n[:, 1, 1] / 1e3),
        "extract_func": lambda X: (X["regression"][:, 1, 1] / 1e3),
        "label": r"$p_{y}(\overline{\nu})$ [GeV]",
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{y}(\overline{\nu})$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{y}(\overline{\nu})$ Deviation [GeV]",
        },
    },
    "nubar_pz": {
        "compute_func": lambda l, j, n: (n[:, 1, 2] / 1e3),
        "extract_func": lambda X: (X["regression"][:, 1, 2] / 1e3),
        "label": r"$p_{z}(\overline{\nu})$ [GeV]",
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{z}(\overline{\nu})$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{z}(\overline{\nu})$ Deviation [GeV]",
        },
    },
    "top_px": {
        "compute_func": lambda l, j, n: (
            TopReconstructor.compute_top_lorentz_vectors(l, j, n)[0][..., 0] / 1e3
        ),
        "extract_func": lambda X: (make_4vect(X["top_truth"][:, 0, :4])[..., 0] / 1e3,),
        "label": r"$p_{x}(t)$ [GeV]",
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{x}(t)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{x}(t)$ Deviation [GeV]",
        },
    },
    "top_py": {
        "compute_func": lambda l, j, n: (
            TopReconstructor.compute_top_lorentz_vectors(l, j, n)[0][..., 1] / 1e3
        ),
        "extract_func": lambda X: (make_4vect(X["top_truth"][:, 0, :4])[..., 1] / 1e3,),
        "label": r"$p_{y}(t)$ [GeV]",
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{y}(t)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{y}(t)$ Deviation [GeV]",
        },
    },
    "top_pz": {
        "compute_func": lambda l, j, n: (
            TopReconstructor.compute_top_lorentz_vectors(l, j, n)[0][..., 2] / 1e3
        ),
        "extract_func": lambda X: (make_4vect(X["top_truth"][:, 0, :4])[..., 2] / 1e3,),
        "label": r"$p_{z}(t)$ [GeV]",
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{z}(t)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{z}(t)$ Deviation [GeV]",
        },
    },
    "top_energy": {
        "compute_func": lambda l, j, n: (
            TopReconstructor.compute_top_lorentz_vectors(l, j, n)[0][..., 3] / 1e3,
        ),
        "extract_func": lambda X: (make_4vect(X["top_truth"][:, 0, :4])[..., 3] / 1e3,),
        "label": r"$E(t)$ [GeV]",
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $E(t)$ Resolution",
            "ylabel_deviation": r"Mean Relative $E(t)$ Deviation",
        },
    },
    "top_pt": {
        "compute_func": lambda l, j, n: (
            compute_pt_from_lorentz_vector_array(
                TopReconstructor.compute_top_lorentz_vectors(l, j, n)[0]
            )
            / 1e3
        ),
        "extract_func": lambda X: (
            compute_pt_from_lorentz_vector_array(make_4vect(X["top_truth"][:, 0, :4]))
            / 1e3
        ),
        "label": r"$p_{T}(t)$ [GeV]",
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $p_{T}(t)$ Resolution",
            "ylabel_deviation": r"Mean Relative $p_{T}(t)$ Deviation",
        },
    },
    "top_gamma": {
        "compute_func": lambda l, j, n: (
            TopReconstructor.compute_top_lorentz_vectors(l, j, n)[0][..., 3]
            / compute_mass_from_lorentz_vector_array(
                TopReconstructor.compute_top_lorentz_vectors(l, j, n)[0]
            )
        ),
        "extract_func": lambda X: (
            make_4vect(X["top_truth"][:, 0, :4])[..., 3]
            / compute_mass_from_lorentz_vector_array(
                make_4vect(X["top_truth"][:, 0, :4])
            )
        ),
        "label": r"$\gamma(t)$",
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$\gamma(t)$ Resolution",
            "ylabel_deviation": r"Mean $\gamma(t)$ Deviation",
        },
    },
    "W_mass": {
        "compute_func": lambda l, j, n: (
            compute_mass_from_lorentz_vector_array((l[:, 0, :4]) + (n[:, 0, :])) / 1e3
        ),
        "extract_func": lambda X: (
            compute_mass_from_lorentz_vector_array(
                make_4vect(X["lep_inputs"][:, 0, :4])
                + make_nu_4vect(X["regression"][:, 0, :])
            )
            / 1e3
        ),
        "label": r"$m(W)$ [GeV]",
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$m(W)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $m{x}(W)$ Deviation [GeV]",
        },
    },
    "W_energy": {
        "compute_func": lambda l, j, n: (
            ((l[:, 0, :4])[..., 3] + (n[:, 0, :])[..., 3]) / 1e3
        ),
        "extract_func": lambda X: (
            (
                make_4vect(X["lep_inputs"][:, 0, :4])[..., 3]
                + make_nu_4vect(X["regression"][:, 0, :])[..., 3]
            )
            / 1e3
        ),
        "label": r"$E(W)$ [GeV]",
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $E(W)$ Resolution",
            "ylabel_deviation": r"Mean Relative $E(W)$ Deviation",
        },
    },
    "W_pt": {
        "compute_func": lambda l, j, n: (
            compute_pt_from_lorentz_vector_array((l[:, 0, :4]) + (n[:, 0, :])) / 1e3
        ),
        "extract_func": lambda X: (
            compute_pt_from_lorentz_vector_array(
                make_4vect(X["lep_inputs"][:, 0, :4])
                + make_nu_4vect(X["regression"][:, 0, :])
            )
            / 1e3
        ),
        "label": r"$p_{T}(W)$ [GeV]",
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $p_{T}(W)$ Resolution",
            "ylabel_deviation": r"Mean Relative $p_{T}(W)$ Deviation",
        },
    },
}
