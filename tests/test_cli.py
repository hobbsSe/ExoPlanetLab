"""The launcher epl.py: configuration file, command line, checks of names."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import epl  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _args(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('config', nargs='?')
    ap.add_argument('--set', action='append')
    for flag, name in epl.SHORTCUTS:
        ap.add_argument(flag, dest=name.lower(), default=None)
    return ap.parse_args(argv)


def test_known_settings_cover_the_core_ones():
    known = epl.known_settings()
    for name in ('SOURCE_MODE', 'SOURCE_ID', 'SOLVER', 'RV_MODE', 'N_FIT', 'EVIDENCE', 'SEED',
                 'RUN_TAG', 'ASTRO_NOISE', 'INJECT', 'RV_FILE', 'M_STAR_MSUN', 'PARALLAX_MAS',
                 'M_STAR_SIGMA', 'PARALLAX_SIGMA', 'SEED_RANK', 'START_SEARCH',
                 'PT_STORE_MAX_GB', 'DR4_PRERELEASE_FILE'):
        assert name in known, name


@pytest.mark.parametrize('value, expected', [
    (True, '1'), (False, '0'), (3, '3'), (4318465066420528000, '4318465066420528000'),
    (0.3, '0.3'), ('emcee', 'emcee'), ([500, 0.3, 0.4, 60, 30, 90], '500,0.3,0.4,60,30,90')])
def test_to_env_value(value, expected):
    assert epl.to_env_value(value) == expected


def test_tables_and_nested_lists_are_rejected():
    with pytest.raises(ValueError):
        epl.to_env_value({'a': 1})
    with pytest.raises(ValueError):
        epl.to_env_value([[1, 2], [3]])


def test_normalise():
    assert epl.normalise('source_id') == 'SOURCE_ID'
    assert epl.normalise('EXOPLANETLAB_SOLVER') == 'SOLVER'
    assert epl.normalise('rv-mode') == 'RV_MODE'


def test_config_file_and_precedence(tmp_path):
    cfg = tmp_path / 'run.toml'
    cfg.write_text('source_id = 1457486023639239296\nsolver = "emcee"\nevidence = true\n'
                   'inject = [500, 0.3, 0.4, 60, 30, 90]\n')
    known = epl.known_settings()
    s = epl.build_settings(_args([str(cfg), '--set', 'solver=dynesty', '--tag', 'X']), known)
    assert s == {'SOURCE_ID': '1457486023639239296', 'SOLVER': 'dynesty', 'EVIDENCE': '1',
                 'INJECT': '500,0.3,0.4,60,30,90', 'RUN_TAG': 'X'}


def test_unknown_setting_is_an_error(tmp_path, capsys):
    cfg = tmp_path / 'run.toml'
    cfg.write_text('sovler = "emcee"\n')
    with pytest.raises(SystemExit) as ex:
        epl.build_settings(_args([str(cfg)]), epl.known_settings())
    assert 'sovler' in str(ex.value) and 'solver' in str(ex.value)
    with pytest.raises(SystemExit):
        epl.build_settings(_args(['--set', 'not_a_setting=1']), epl.known_settings())


def test_example_configurations_are_valid():
    known = epl.known_settings()
    examples = [f for f in os.listdir(os.path.join(ROOT, 'examples')) if f.endswith('.toml')]
    assert examples
    for f in examples:
        s = epl.build_settings(_args([os.path.join(ROOT, 'examples', f)]), known)
        assert 'SOURCE_MODE' in s


def test_dry_run_and_list(capsys):
    assert epl.main([os.path.join(ROOT, 'examples', 'gaia4.toml'), '--dry-run']) == 0
    assert 'source_id' in capsys.readouterr().out
    assert epl.main(['--list-settings']) == 0
    assert 'solver' in capsys.readouterr().out
