import pytest
from click.testing import CliRunner

from distripute.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_info_no_master(self, runner):
        result = runner.invoke(cli, ["info"])
        # fails because no master running — but should not crash
        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "master" in result.output
        assert "worker" in result.output
        assert "relay" in result.output
        assert "job" in result.output

    def test_job_help(self, runner):
        result = runner.invoke(cli, ["job", "--help"])
        assert result.exit_code == 0
        assert "create" in result.output
        assert "list" in result.output
        assert "status" in result.output

    def test_worker_help(self, runner):
        result = runner.invoke(cli, ["workers", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output

    def test_model_help(self, runner):
        result = runner.invoke(cli, ["model", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output or "add" in result.output
