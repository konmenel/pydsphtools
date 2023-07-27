#!/usr/bin/env python3
"""Copyright (C) 2023 Constantinos Menelaou <https://github.com/konmenel>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

@author: Constantinos Menelaou
@github: https://github.com/konmenel
@year: 2023
"""
import os
import numpy as np
import pandas as pd

from pydsphtools import *

NUMBER_OF_TESTS = 12


def testing() -> int:
    test_dir, _ = os.path.split(__file__)

    # Test build-in types
    assert get_dp(test_dir) == 0.003
    assert get_usr_def_var(test_dir, "BoulderRho", int) == 2800
    assert get_usr_def_var(test_dir, "BoulderVol", float) == -6.75e-06
    assert get_chrono_mass(test_dir, "boulder") == 0.02646
    assert get_chrono_property(test_dir, "boulder", "MkBound") == 50
    assert get_chrono_property(test_dir, "boulder", "Mass") == 0.02646
    assert get_chrono_property(test_dir, "boulder", "ModelFile") == "boulder_mkb0050.obj"

    # Test numpy arrays
    assert np.array_equal(
        get_chrono_inertia(test_dir, "boulder"),
        np.array([2.44094e-06, 1.42884e-06, 2.91722e-06]),
    )
    assert np.array_equal(
        get_chrono_property(test_dir, "boulder", "Center"),
        np.array([-0.0325, 0.1515, 0.0135]),
    )
    assert np.array_equal(
        get_chrono_property(test_dir, "boulder", "Inertia"),
        np.array([2.44094e-06, 1.42884e-06, 2.91722e-06]),
    )

    # Test StringIO stream
    stream = read_and_fix_csv(test_dir)
    df = pd.read_csv(stream, sep=";")
    CONFIGURATION = (
        "CaseKfric46 - 3D - mDBC(DBC vel=0 - FastSingle) - Symplectic"
        + " - Wendland - Visco_Laminar+SPS(1e-06b1) - DDT3(0.1)"
        + " - Shifting(NoFixed:-2:2.75:Full) - Ft-CHRONO - RhopOut(700-1300)"
    )
    assert df["Configuration"][0] == CONFIGURATION

    print("All test were successful!")
    return 0


if __name__ == "__main__":
    raise SystemExit(testing())
