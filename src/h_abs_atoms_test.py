import os
import sys
import unittest

import pandas as pd
from arc.species.converter import str_to_xyz

from h_abs_atoms import convert_xyz_to_df, get_h_abs_atoms, pull_atoms_closer, get_element_symbol

arc_path = "~/code/ARC"
rmg_path = "~/code/RMG-Py"
rmg_database_path = "~/code/RMG-database"
sys.path.append(os.path.expanduser((arc_path)))
sys.path.append(os.path.expanduser((rmg_path)))


class TestHAbAtom(unittest.TestCase):
    """
    Contains unit tests for identifying the molecules that are partaking in the hydrogen abstraction reaction
    """
    def test_get_element_symbol(self):
        """
        Test the extraction of the element symbol from an atom label
        """
        self.assertEqual(get_element_symbol("C0"), "C")
        self.assertEqual(get_element_symbol("C"), "C")
        self.assertEqual(get_element_symbol("H"), "H")
        self.assertEqual(get_element_symbol("H1"), "H")
        self.assertEqual(get_element_symbol("O2"), "O")
        self.assertEqual(get_element_symbol("O"), "O")
        self.assertEqual(get_element_symbol("S14"), "S")


    def test_convert_xyz_to_df(self):
        """
        Test the conversion of the xyz dict to a pandas dataframe
        """

        ### Transition State 1 Test

        ts_1 = dict(
            symbols=(
                "C",
                "C",
                "O",
                "C",
                "O",
                "C",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "C",
                "C",
                "S",
                "C",
                "C",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
            ),
            isotopes=(
                12,
                12,
                16,
                12,
                16,
                12,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                12,
                12,
                32,
                12,
                12,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ),
            coords=(
                (-0.092214, -1.066792, -0.031002),
                (0.095866, -0.432812, 1.320494),
                (-0.514445, -1.236626, 2.326523),
                (-1.858508, -1.468562, 2.036188),
                (-2.050671, -2.150886, 0.833065),
                (-1.542292, -1.397099, -0.262601),
                (0.581313, -1.899984, -0.225328),
                (1.144751, -0.335404, 1.591683),
                (-0.354598, 0.57127, 1.335694),
                (-2.398774, -0.504523, 2.000083),
                (-2.256965, -2.096994, 2.829993),
                (-2.132669, -0.473867, -0.362693),
                (-1.695727, -2.002163, -1.154944),
                (-0.814285, 1.512421, -2.090166),
                (0.532527, 0.898542, -1.78603),
                (1.653107, 2.050769, -1.029154),
                (3.238984, 1.210729, -1.257302),
                (3.425404, -0.049485, -0.429013),
                (-0.732121, 2.314746, -2.82686),
                (-1.492258, 0.754871, -2.486826),
                (-1.268058, 1.930374, -1.188843),
                (0.292293, -0.072922, -0.968424),
                (0.989576, 0.413048, -2.650172),
                (3.361833, 1.002207, -2.322074),
                (3.985463, 1.959881, -0.989153),
                (2.681984, -0.80514, -0.684225),
                (3.337107, 0.1684, 0.635723),
                (4.413885, -0.477249, -0.610195),
            ),
        )

        expected_series = pd.Series(
            {
                "C0": 0.000000,
                "C1": 1.504608,
                "O2": 2.401051,
                "C3": 2.748543,
                "O4": 2.399464,
                "C5": 1.505147,
                "H6": 1.088857,
                "H7": 2.167514,
                "H8": 2.149407,
                "H9": 3.124367,
                "H10": 3.732661,
                "H11": 2.150590,
                "H12": 2.170119,
                "C13": 3.378444,
                "C14": 2.707944,
                "S15": 3.709668,
                "C16": 4.217558,
                "C17": 3.683336,
                "H18": 4.434084,
                "H19": 3.362982,
                "H20": 3.421434,
                "H21": 1.419290,
                "H22": 3.196912,
                "H23": 4.632516,
                "H24": 5.167809,
                "H25": 2.862051,
                "H26": 3.705464,
                "H27": 4.581261,
            }
        )
        ts_1_df = convert_xyz_to_df(ts_1)
        result_series = ts_1_df.iloc[0]
        for label in expected_series.index:
            self.assertAlmostEqual(
                result_series[label], expected_series[label], places=6
            )

        self.assertEqual(
            tuple(ts_1_df.columns),
            (
                "C0",
                "C1",
                "O2",
                "C3",
                "O4",
                "C5",
                "H6",
                "H7",
                "H8",
                "H9",
                "H10",
                "H11",
                "H12",
                "C13",
                "C14",
                "S15",
                "C16",
                "C17",
                "H18",
                "H19",
                "H20",
                "H21",
                "H22",
                "H23",
                "H24",
                "H25",
                "H26",
                "H27",
            ),
        )
        self.assertEqual(
            tuple(ts_1_df.index),
            (
                "C0",
                "C1",
                "O2",
                "C3",
                "O4",
                "C5",
                "H6",
                "H7",
                "H8",
                "H9",
                "H10",
                "H11",
                "H12",
                "C13",
                "C14",
                "S15",
                "C16",
                "C17",
                "H18",
                "H19",
                "H20",
                "H21",
                "H22",
                "H23",
                "H24",
                "H25",
                "H26",
                "H27",
            ),
        )

        ### Transition State 2 Test
        ts_2 = [
            "H       0.00000000    0.00000000   -2.62704900",
            "H       0.00000000    0.00000000   -1.25723600",
            "S       0.00000000    0.00000000    0.13445800",
        ]
        ts_2 = "\n".join(ts_2)
        ts_2_xyz = str_to_xyz(ts_2)
        ts_2_df = convert_xyz_to_df(ts_2_xyz)
        expected_data = pd.DataFrame(
            [
                [0.000000, 1.369813, 2.761507],
                [1.369813, 0.000000, 1.391694],
                [2.761507, 1.391694, 0.000000],
            ],
            columns=["H0", "H1", "S2"],  # Define the columns
            index=["H0", "H1", "S2"],  # Define the index
        )
        self.assertTrue(expected_data.equals(ts_2_df))
        self.assertEqual(list(ts_2_df.columns), ["H0", "H1", "S2"])
        self.assertEqual(list(ts_2_df.index), ["H0", "H1", "S2"])

    def test_pull_atoms_closer(self):
        # Your initial data
        test_extracted_xyz_from_ts_hh = {
            "symbols": ("H", "H"),
            "isotopes": (1, 1),
            "coords": ((0.0, 0.0, -0.5651074999999999), (0.0, 0.0, 0.5651075000000001)),
        }
        updated_xyz_hh = pull_atoms_closer(
            test_extracted_xyz_from_ts_hh, h_index=0, a_index=1, target_distance=0.8
        )
        self.assertAlmostEqual(
            updated_xyz_hh["coords"][0], [0.0, 0.0, -0.23489249999999995], places=6
        )

        test_extracted_xyz_from_ts_oh = {
            "symbols": ("H", "O"),
            "isotopes": (1, 16),
            "coords": (
                (-0.06168997099804931, -1.3570636526926665, -0.07330417082332219),
                (0.0038870290019506967, 0.08550734730733356, 0.0046188291766778855),
            ),
        }
        updated_xyz_oh = pull_atoms_closer(
            test_extracted_xyz_from_ts_oh, h_index=0, a_index=1, target_distance=0.8
        )
        self.assertAlmostEqual(
            updated_xyz_oh["coords"][0],
            [-0.0323894143277068, -0.7125063637576334, -0.0384872795745139],
            places=6,
        )

    def test_get_h_abs_atoms_imp(self):
        ts_3 = [
            "Cl      0.00000000    0.00000000    0.07879500",
            "H       0.00000000    0.00000000   -2.42303100",
            "H       0.00000000    0.00000000   -1.29281600",
        ]
        ts_3 = "\n".join(ts_3)
        ts_3_xyz = str_to_xyz(ts_3)
        ts_3_df = convert_xyz_to_df(ts_3_xyz)

        ts_3_results = get_h_abs_atoms(ts_3_df)
        self.assertEqual(ts_3_results, {"H": 2, "A": 1, "B": 0, "C": None, "D": None})

        ts_4 = [
            "H       0.00000000    1.92749700    0.41568400",
            "N       0.00000000   -0.22043100    0.02230400",
            "H       0.00000000    0.88154400    0.37784100",
            "H       0.00000000   -0.09935300   -1.00225600",
        ]
        ts_4 = "\n".join(ts_4)
        ts_4_xyz = str_to_xyz(ts_4)
        ts_4_df = convert_xyz_to_df(ts_4_xyz)
        ts_4_results = get_h_abs_atoms(ts_4_df)
        self.assertEqual(ts_4_results, {"H": 2, "A": 0, "B": 1, "C": None, "D": 3})

        ts_5 = [
            "Cl      0.00000000    0.76620700    0.27089500",
            "C       0.00000000   -1.94865700   -0.68895300",
            "H       0.00000000   -0.57799000   -0.20435000",
            "H       0.92220000   -2.31342000   -0.25318600",
            "H      -0.92220000   -2.31342000   -0.25318600",
            "H       0.00000000   -1.78098800   -1.75913300",
        ]
        ts_5 = "\n".join(ts_5)
        ts_5_xyz = str_to_xyz(ts_5)
        ts_5_df = convert_xyz_to_df(ts_5_xyz)
        ts_5_results = get_h_abs_atoms(ts_5_df)
        self.assertEqual(ts_5_results, {"H": 2, "A": 0, "B": 1, "C": None, "D": 3})

        ts_6 = [
            "C       0.00000000   -0.07902300   -1.21873400",
            "S       0.00000000   -0.11616000    0.58903600",
            "H       0.89314500    0.40735000   -1.60555600",
            "H       0.00000000   -1.11812300   -1.54545800",
            "H      -0.89314500    0.40735000   -1.60555600",
            "H       0.00000000    1.26172300    0.77386300",
            "H       0.00000000    2.58964600    0.73354900",
        ]
        ts_6 = "\n".join(ts_6)
        ts_6_xyz = str_to_xyz(ts_6)
        ts_6_df = convert_xyz_to_df(ts_6_xyz)
        ts_6_results = get_h_abs_atoms(ts_6_df)
        self.assertEqual(ts_6_results, {"H": 5, "A": 6, "B": 1, "C": None, "D": 0})

        ts7 = """C      -0.00395600   -0.04326400   -1.40455000
            C      -0.00845300   -0.09246500   -0.20543800
            C      -0.00943900   -0.10325600    1.22383200
            H      -0.00109100   -0.01190900   -2.46751900
            H       0.09877600    1.08073700    1.63721200
            H       0.85012800   -0.60354300    1.66512700
            H      -0.94545600   -0.43943000    1.66512700
            H       0.18507500    2.02494800    2.01782400
            """
        ts7_xyz = str_to_xyz(ts7)
        ts7_df = convert_xyz_to_df(ts7_xyz)
        ts7_results = get_h_abs_atoms(ts7_df)
        self.assertEqual(ts7_results, {"H": 4, "A": 7, "B": 2, "C": None, "D": 1})

        ts_8 = [
            "C      -0.09221400   -1.06679200   -0.03100200",
            "C       0.09586600   -0.43281200    1.32049400",
            "O      -0.51444500   -1.23662600    2.32652300",
            "C      -1.85850800   -1.46856200    2.03618800",
            "O      -2.05067100   -2.15088600    0.83306500",
            "C      -1.54229200   -1.39709900   -0.26260100",
            "H       0.58131300   -1.89998400   -0.22532800",
            "H       1.14475100   -0.33540400    1.59168300",
            "H      -0.35459800    0.57127000    1.33569400",
            "H      -2.39877400   -0.50452300    2.00008300",
            "H      -2.25696500   -2.09699400    2.82999300",
            "H      -2.13266900   -0.47386700   -0.36269300",
            "H      -1.69572700   -2.00216300   -1.15494400",
            "C      -0.81428500    1.51242100   -2.09016600",
            "C       0.53252700    0.89854200   -1.78603000",
            "S       1.65310700    2.05076900   -1.02915400",
            "C       3.23898400    1.21072900   -1.25730200",
            "C       3.42540400   -0.04948500   -0.42901300",
            "H      -0.73212100    2.31474600   -2.82686000",
            "H      -1.49225800    0.75487100   -2.48682600",
            "H      -1.26805800    1.93037400   -1.18884300",
            "H       0.29229300   -0.07292200   -0.96842400",
            "H       0.98957600    0.41304800   -2.65017200",
            "H       3.36183300    1.00220700   -2.32207400",
            "H       3.98546300    1.95988100   -0.98915300",
            "H       2.68198400   -0.80514000   -0.68422500",
            "H       3.33710700    0.16840000    0.63572300",
            "H       4.41388500   -0.47724900   -0.61019500",
        ]
        ts_8 = "\n".join(ts_8)
        ts_8_xyz = str_to_xyz(ts_8)
        ts_8_df = convert_xyz_to_df(ts_8_xyz)
        ts_8_results = get_h_abs_atoms(ts_8_df)
        self.assertEqual(ts_8_results, {"H": 21, "A": 14, "B": 0, "C": 15, "D": 1})

        ts_89 = [
            "C      -0.00362700   -0.77718600   -1.12561500",
            "N      -0.04947600   -0.87252000    0.12257000",
            "H       0.05895900    0.17226200   -1.66069600",
            "H      -0.02742400   -1.69437600   -1.71552500",
            "H      -0.02200400    0.07419100    0.64372000",
            "O       0.04357300    1.51676200    0.72164300",
        ]
        ts_89 = "\n".join(ts_89)
        ts_89_xyz = str_to_xyz(ts_89)
        ts_89_df = convert_xyz_to_df(ts_89_xyz)
        ts_89_results = get_h_abs_atoms(ts_89_df)
        self.assertEqual(ts_89_results, {"H": 4, "A": 1, "B": 5, "C": 0, "D": None})
