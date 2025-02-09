import unittest

from arc.species.converter import check_xyz_dict

import h_abs_atoms
import perception
import split


class PerceptionTest(unittest.TestCase):

    def test_perception(self):
        """
        Test the perception module
        """
        original_r2h_xyz = [
            "C      -2.05538800    0.07234800   -0.49099900",
            "C      -0.74610400   -0.62663400   -0.81778700",
            "S       0.32493500    0.25905800   -1.97823800",
            "C      -0.71718300    0.36419500   -3.45511200",
            "C      -1.13746000   -0.97365000   -4.04088400",
            "H      -2.70046000    0.14267700   -1.36842600",
            "H      -2.60160100   -0.48482600    0.27334200",
            "H      -1.87582500    1.08250800   -0.12132600",
            "H      -0.13198500   -0.72455400    0.07893700",
            "H      -0.92537100   -1.63744800   -1.18785600",
            "H      -1.58731400    0.98738400   -3.24217900",
            "H      -0.10212300    0.91614700   -4.16777900",
            "H      -1.78758900   -1.52239800   -3.35739900",
            "H      -0.26763400   -1.59618800   -4.25342200",
            "H      -1.69158400   -0.82291700   -4.96998900",
        ]
        original_r2h_xyz = "\n".join(original_r2h_xyz)
        original_r2h_xyz = check_xyz_dict(original_r2h_xyz)

        original_r1h_xyz = [
            "C      -1.17782200    0.22414900    0.18179300",
            "C      -0.56332800   -1.16364800    0.06776700",
            "O       0.69432400   -1.19963000    0.71463000",
            "C       1.55474000   -0.22884200    0.20113300",
            "O       1.07294000    1.06956400    0.36976300",
            "C      -0.16072200    1.24932200   -0.29895000",
            "H      -2.09274500    0.28660800   -0.41169100",
            "H      -1.43029900    0.42478700    1.22478600",
            "H      -1.18219600   -1.92346200    0.54307400",
            "H      -0.43892400   -1.43466100   -0.99190600",
            "H       1.72499500   -0.42067700   -0.87421200",
            "H       2.48963500   -0.30105000    0.75239500",
            "H      -0.00799200    1.14808000   -1.38442500",
            "H      -0.48260400    2.26946000   -0.09415600",
        ]
        original_r1h_xyz = "\n".join(original_r1h_xyz)
        original_r1h_xyz = check_xyz_dict(original_r1h_xyz)

        original_ts_xyz = [
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
        original_ts_xyz = "\n".join(original_ts_xyz)
        original_ts_xyz = check_xyz_dict(original_ts_xyz)
        ts_df = h_abs_atoms.convert_xyz_to_df(original_ts_xyz)
        ts_results = h_abs_atoms.get_h_abs_atoms(ts_df)

        g1_xyz, g2_xyz, index_dict = split.get_group_xyzs_and_key_indices_from_ts(
            xyz=original_ts_xyz, a=ts_results["A"], b=ts_results["B"], h=ts_results["H"]
        )
        self.assertEqual(
            g1_xyz,
            {
                "symbols": (
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
                "isotopes": (12, 12, 32, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                "coords": (
                    (-2.426009974283668, 0.22782502599993149, -0.825810393516299),
                    (-1.079197974283668, -0.3860539740000686, -0.5216743935162991),
                    (0.041382025716332205, 0.7661730259999313, 0.23520160648370103),
                    (1.627259025716332, -0.07386697400006859, 0.007053606483701014),
                    (1.813679025716332, -1.3340809740000685, 0.835342606483701),
                    (-2.343845974283668, 1.0301500259999314, -1.562504393516299),
                    (-3.103982974283668, -0.5297249740000686, -1.2224703935162993),
                    (-2.8797829742836676, 0.6457780259999315, 0.07551260648370084),
                    (-1.3194319742836678, -1.3575179740000685, 0.295931606483701),
                    (-0.6221489742836679, -0.8715479740000684, -1.385816393516299),
                    (1.750108025716332, -0.28238897400006846, -1.0577183935162993),
                    (2.3737380257163325, 0.6752850259999315, 0.275202606483701),
                    (1.070259025716332, -2.0897359740000683, 0.580130606483701),
                    (1.7253820257163321, -1.1161959740000684, 1.900078606483701),
                    (2.802160025716332, -1.7618449740000686, 0.6541606064837009),
                ),
            },
        )
        self.assertEqual(
            g2_xyz,
            {
                "symbols": (
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
                    "H",
                ),
                "isotopes": (12, 12, 16, 12, 16, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                "coords": (
                    (0.9147807368612055, 0.22146384056750534, -1.080147814989551),
                    (1.1028607368612056, 0.8554438405675053, 0.2713481850104491),
                    (0.4925497368612055, 0.0516298405675053, 1.277377185010449),
                    (-0.8515132631387945, -0.18030615943249462, 0.9870421850104492),
                    (-1.0436762631387944, -0.8626301594324945, -0.2160808149895509),
                    (-0.5352972631387944, -0.10884315943249478, -1.3117468149895508),
                    (1.5883077368612055, -0.6117281594324946, -1.274473814989551),
                    (2.151745736861206, 0.9528518405675053, 0.542537185010449),
                    (0.6523967368612056, 1.8595258405675053, 0.286548185010449),
                    (-1.3917792631387944, 0.7837328405675053, 0.9509371850104491),
                    (-1.2499702631387946, -0.8087381594324947, 1.780847185010449),
                    (-1.1256742631387944, 0.8143888405675053, -1.4118388149895509),
                    (-0.6887322631387944, -0.7139071594324946, -2.204089814989551),
                    (1.2992877368612055, 1.2153338405675054, -2.017569814989551),
                ),
            },
        )
        self.assertEqual(index_dict, {"g1_a": 1, "g1_h": 8, "g2_a": 0, "g2_h": 13})

        iso_r2h = perception.check_isomorphism(g1_xyz, original_r2h_xyz)
        self.assertTrue(iso_r2h)
        iso_r1h = perception.check_isomorphism(g2_xyz, original_r1h_xyz)
        self.assertTrue(iso_r1h)

        original_r2h_xyz_1 = [
            "Cl      0.00000000    0.00000000    0.63913800",
            "H       0.00000000    0.00000000   -0.63913800",
        ]
        original_r2h_xyz_1 = "\n".join(original_r2h_xyz_1)
        original_r2h_xyz_1 = check_xyz_dict(original_r2h_xyz_1)

        original_r1h_xyz_1 = [
            "H       0.00000000    0.00000000    0.37207800",
            "H       0.00000000    0.00000000   -0.37207800",
        ]
        original_r1h_xyz_1 = "\n".join(original_r1h_xyz_1)
        original_r1h_xyz_1 = check_xyz_dict(original_r1h_xyz_1)

        original_ts_xyz_1 = [
            "Cl      0.00000000    0.00000000    0.07879500",
            "H       0.00000000    0.00000000   -2.42303100",
            "H       0.00000000    0.00000000   -1.29281600",
        ]
        original_ts_xyz_1 = "\n".join(original_ts_xyz_1)
        original_ts_xyz_1 = check_xyz_dict(original_ts_xyz_1)
        ts_df_1 = h_abs_atoms.convert_xyz_to_df(original_ts_xyz_1)
        ts_results_1 = h_abs_atoms.get_h_abs_atoms(ts_df_1)

        g1_xyz_1, g2_xyz_1, index_dict_1 = split.get_group_xyzs_and_key_indices_from_ts(
            xyz=original_ts_xyz_1,
            a=ts_results_1["A"],
            b=ts_results_1["B"],
            h=ts_results_1["H"],
        )

        self.assertEqual(
            g2_xyz_1,
            {
                "symbols": ("Cl", "H"),
                "isotopes": (35, 1),
                "coords": (
                    (0.0, 0.0, 0.03842333389635337),
                    (0.0, 0.0, -1.3331876661036466),
                ),
            },
        )
        self.assertEqual(
            g1_xyz_1,
            {
                "symbols": ("H", "H"),
                "isotopes": (1, 1),
                "coords": (
                    (0.0, 0.0, -0.5651074999999999),
                    (0.0, 0.0, 0.5651075000000001),
                ),
            },
        )
        self.assertEqual(index_dict_1, {"g1_a": 0, "g1_h": 1, "g2_a": 0, "g2_h": 1})

        iso_r2h_1 = perception.check_isomorphism(g2_xyz_1, original_r2h_xyz_1)
        self.assertTrue(iso_r2h_1)
        iso_r1h_1 = perception.check_isomorphism(h_abs_atoms.pull_atoms_closer(g1_xyz_1, h_index=index_dict_1["g1_h"],a_index=index_dict_1["g1_a"]), original_r1h_xyz_1)
        self.assertTrue(iso_r1h_1)

        original_r1h_xyz_2 = [
            "O       0.00000000    0.00000000    0.48548100",
            "H       0.00000000    0.00000000   -0.48548100",
        ]
        original_r1h_xyz_2 = "\n".join(original_r1h_xyz_2)
        original_r1h_xyz_2 = check_xyz_dict(original_r1h_xyz_2)

        original_r2h_xyz_2 = [
            "C      -0.36888300   -0.05399500    0.11836100",
            "N       0.78618000    0.43592900    0.23941200",
            "H      -0.68716400   -0.70345100   -0.70333800",
            "H      -1.11935400    0.17332000    0.87582800",
            "H       1.38922000    0.14819800   -0.53026200",
        ]
        original_r2h_xyz_2 = "\n".join(original_r2h_xyz_2)
        original_r2h_xyz_2 = check_xyz_dict(original_r2h_xyz_2)

        original_ts_xyz_2 = [
            "C      -0.00362700   -0.77718600   -1.12561500",
            "N      -0.04947600   -0.87252000    0.12257000",
            "H       0.05895900    0.17226200   -1.66069600",
            "H      -0.02742400   -1.69437600   -1.71552500",
            "H      -0.02200400    0.07419100    0.64372000",
            "O       0.04357300    1.51676200    0.72164300",
        ]
        original_ts_xyz_2 = "\n".join(original_ts_xyz_2)
        original_ts_xyz_2 = check_xyz_dict(original_ts_xyz_2)
        ts_df_2 = h_abs_atoms.convert_xyz_to_df(original_ts_xyz_2)
        ts_results_2 = h_abs_atoms.get_h_abs_atoms(ts_df_2)

        g1_xyz_2, g2_xyz_2, index_dict_2 = split.get_group_xyzs_and_key_indices_from_ts(
            xyz=original_ts_xyz_2,
            a=ts_results_2["A"],
            b=ts_results_2["B"],
            h=ts_results_2["H"],
        )
        iso_r2h_1 = perception.check_isomorphism(
            h_abs_atoms.pull_atoms_closer(
                g1_xyz_2,
                h_index=index_dict_2["g1_h"],
                a_index=index_dict_2["g1_a"],
                target_distance=0.8,
            ),
            original_r2h_xyz_2,
        )
        iso_r2h_2 = perception.check_isomorphism(
            h_abs_atoms.pull_atoms_closer(
                g2_xyz_2,
                h_index=index_dict_2["g2_h"],
                a_index=index_dict_2["g2_a"],
                target_distance=0.8,
            ),
            original_r1h_xyz_2,
        )
        self.assertTrue(iso_r2h_1)
        self.assertTrue(iso_r2h_2)
