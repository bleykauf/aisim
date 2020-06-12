import pytest  # noqa
import aisim as ais


def test_wavefront():
    # test the methods of Wavefront that are not covered by wf_test.py
    wf = ais.gen_wavefront(1e-3)
    wf.plot()
    wf.plot_coeff()
