#!/usr/bin/env python3
# encoding: utf-8

import unittest

import split as split
from arc.species.converter import check_xyz_dict, xyz_to_dmat




class TestTSSplit(unittest.TestCase):
    """
    Contains unit tests for ARC's split module
    """
    maxDiff = None

    def test_bonded(self):
        """Test bonded"""
        self.assertTrue(split.bonded(1.0, 'C', 'C'))
        self.assertTrue(split.bonded(1.0, 'C', 'H'))
        self.assertTrue(split.bonded(1.0, 'H', 'C'))
        self.assertTrue(split.bonded(1.0, 'H', 'H'))
        self.assertFalse(split.bonded(1.7, 'C', 'C'))
        self.assertFalse(split.bonded(1.5, 'C', 'H'))
        self.assertFalse(split.bonded(1.5, 'H', 'C'))
        self.assertFalse(split.bonded(1.5, 'H', 'H'))

    def test_get_adjlist_from_dmat(self):
        """Test get_adjlist_from_dmat"""

        ts_coords = """C      -0.00395600   -0.04326400   -1.40455000
C      -0.00845300   -0.09246500   -0.20543800
C      -0.00943900   -0.10325600    1.22383200
H      -0.00109100   -0.01190900   -2.46751900
H       0.09877600    1.08073700    1.63721200
H       0.85012800   -0.60354300    1.66512700
H      -0.94545600   -0.43943000    1.66512700
H       0.18507500    2.02494800    2.01782400
"""
        ts_xyz_dict = check_xyz_dict(ts_coords)
        symbols = ts_xyz_dict['symbols']
        ts_dmat = xyz_to_dmat(ts_xyz_dict)
        adjlist = split.get_adjlist_from_dmat(dmat=ts_dmat, symbols=symbols, h=4, a=7, b=2) # b is incorrect chemically
        self.assertEqual(adjlist, {0: [1, 3], 1: [0, 2], 2: [1, 5, 6, 4], 3: [0], 5: [2], 6: [2], 4: [7, 2], 7: [4]})

        # adjlist_opt = split.optimized_get_adjlist_from_dmat(dmat=ts_dmat, symbols=symbols, h=4, a=7, b=2)
        # self.assertEqual(adjlist_opt, adjlist)

        xyz = """ C                 -3.80799396    1.05904061    0.12143410
                  H                 -3.75776386    0.09672979   -0.34366835
                  H                 -3.24934849    1.76454448   -0.45741718
                  H                 -4.82886508    1.37420677    0.17961125
                  C                 -3.21502590    0.97505234    1.54021348
                  H                 -3.26525696    1.93736874    2.00531040
                  H                 -3.77366533    0.26954471    2.11907272
                  C                 -1.74572494    0.52144864    1.45646938
                  H                 -1.18708880    1.22694232    0.87759054
                  H                 -1.69550074   -0.44087971    0.99139265
                  O                 -0.57307243    0.35560699    4.26172088
                  H                 -1.12770789    0.43395779    2.93512192
                  O                  0.45489302    1.17807207    4.35811043
                  H                  1.12427554    0.93029226    3.71613651"""
        xyz_dict = check_xyz_dict(xyz)
        dmat = xyz_to_dmat(xyz_dict)
        adjlist = split.get_adjlist_from_dmat(dmat=dmat, symbols=xyz_dict['symbols'], h=11, a=7, b=10)
        self.assertEqual(adjlist,
                         {0: [1, 2, 3, 4],
                          1: [0],
                          2: [0],
                          3: [0],
                          4: [0, 5, 6, 7],
                          5: [4],
                          6: [4],
                          7: [4, 8, 9, 11],
                          8: [7],
                          9: [7],
                          10: [12, 11],
                          12: [10, 13],
                          13: [12],
                          11: [7, 10]})
      


    def test_iterative_dfs(self):
        """Test iterative_dfs"""
        adjlist = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0, 5, 6, 7], 5: [4], 6: [4], 7: [4, 8, 9, 11], 8: [7],
                   9: [7], 10: [12, 11], 12: [10, 13], 13: [12], 11: [7, 10]}
        g1 = split.iterative_dfs(adjlist=adjlist, start=10, border=11)
        self.assertEqual(g1, [11, 10, 12, 13])
        g2 = split.iterative_dfs(adjlist=adjlist, start=7, border=11)
        self.assertEqual(g2, [11, 7, 9, 8, 4, 6, 5, 0, 3, 2, 1])


    def test_get_group_xyzs_and_key_indices_from_ts_xyz(self):
        """Test get_group_xyzs_and_key_indices_from_ts_xyz"""
        xyz = """ C                 -3.80799396    1.05904061    0.12143410
                  H                 -3.75776386    0.09672979   -0.34366835
                  H                 -3.24934849    1.76454448   -0.45741718
                  H                 -4.82886508    1.37420677    0.17961125
                  C                 -3.21502590    0.97505234    1.54021348
                  H                 -3.26525696    1.93736874    2.00531040
                  H                 -3.77366533    0.26954471    2.11907272
                  C                 -1.74572494    0.52144864    1.45646938
                  H                 -1.18708880    1.22694232    0.87759054
                  H                 -1.69550074   -0.44087971    0.99139265
                  O                 -0.57307243    0.35560699    4.26172088
                  H                 -1.12770789    0.43395779    2.93512192
                  O                  0.45489302    1.17807207    4.35811043
                  H                  1.12427554    0.93029226    3.71613651"""
        a, b, h = 7, 10, 11
        xyz_dict = check_xyz_dict(xyz)
        g1_xyz, g2_xyz, index_dict = split.get_group_xyzs_and_key_indices_from_ts(xyz=xyz_dict, a=a, b=b, h=h)

        self.assertEqual(g1_xyz,  {'symbols': ('C', 'H', 'H', 'H', 'C', 'H', 'H', 'C', 'H', 'H', 'H'), 'isotopes': (12, 1, 1, 1, 12, 1, 1, 12, 1, 1, 1), 'coords': ((-0.8964723656188727, 0.21067833990016682, -0.9177560488779365), (-0.8462422656188724, -0.7516324800998332, -1.3828584988779364), (-0.3378268956188726, 0.9161822099001669, -1.4966073288779365), (-1.9173434856188725, 0.5258444999001668, -0.8595788988779365), (-0.30350430561887265, 0.12669006990016674, 0.5010233311220635), (-0.3537353656188724, 1.0890064699001667, 0.9661202511220635), (-0.8621437356188726, -0.5788175600998332, 1.0798825711220637), (1.1657966543811276, -0.3269136300998332, 0.41727923112206344), (1.7244327943811275, 0.3785800499001668, -0.16159960887793645), (1.2160208543811275, -1.2892419800998332, -0.047797498877936495), (1.7838137043811275, -0.4144044800998332, 1.8959317711220633))})
        self.assertEqual(g2_xyz, {'symbols': ('O', 'H', 'O', 'H'), 'isotopes': (16, 1, 16, 1), 'coords': ((-0.5173834992814444, -0.40621114930172336, 0.010148088176216596), (-1.0720189592814444, -0.32786034930172336, -1.316450871823784), (0.5105819507185555, 0.41625393069827665, 0.1065376381762162), (1.1799644707185555, 0.1684741206982766, -0.5354362818237837))})
        self.assertEqual(index_dict, {'g1_a': 7, 'g1_h': 10, 'g2_a': 0, 'g2_h': 1})


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
