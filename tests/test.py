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
from pydsphtools import *

NUMBER_OF_TESTS = 12


def testing() -> int:
    dirout = "test"

    # Test build-in types
    assert get_dp(dirout) == 0.003
    assert get_usr_def_var(dirout, "BoulderRho", int) == 2800
    assert get_usr_def_var(dirout, "BoulderVol", float) == -6.75e-06
    assert get_chrono_mass(dirout, "boulder") == 0.02646
    assert get_chrono_property(dirout, "boulder", "MkBound") == 50
    assert get_chrono_property(dirout, "boulder", "Mass") == 0.02646
    assert get_chrono_property(dirout, "boulder", "ModelFile") == "boulder_mkb0050.obj"

    # Test numpy arrays
    assert np.array_equal(
        get_chrono_inertia(dirout, "boulder"),
        np.array([2.44094e-06, 1.42884e-06, 2.91722e-06]),
    )
    assert np.array_equal(
        get_chrono_property(dirout, "boulder", "Center"),
        np.array([-0.0325, 0.1515, 0.0135]),
    )
    assert np.array_equal(
        get_chrono_property(dirout, "boulder", "Inertia"),
        np.array([2.44094e-06, 1.42884e-06, 2.91722e-06]),
    )

    # Test StringIO stream
    stream = read_and_fix_csv(dirout)
    df = pd.read_csv(stream, sep=";")
    CONFIGURATION = (
        "CaseKfric46 - 3D - mDBC(DBC vel=0 - FastSingle) - Symplectic"
        + " - Wendland - Visco_Laminar+SPS(1e-06b1) - DDT3(0.1)"
        + " - Shifting(NoFixed:-2:2.75:Full) - Ft-CHRONO - RhopOut(700-1300)"
    )
    assert df["Configuration"][0] == CONFIGURATION

    return 0


if __name__ == "__main__":
    raise SystemExit(testing())
