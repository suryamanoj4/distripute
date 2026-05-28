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

    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "master" in result.output
        assert "worker" in result.output
        assert "relay" in result.output
        assert "info" in result.output

    def test_master_help(self, runner):
        result = runner.invoke(cli, ["master", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--relay" in result.output

    def test_worker_help(self, runner):
        result = runner.invoke(cli, ["worker", "--help"])
        assert result.exit_code == 0
        assert "--network-id" in result.output
        assert "--master" in result.output
        assert "--relay" in result.output

    def test_relay_help(self, runner):
        result = runner.invoke(cli, ["relay", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output

    def test_info_no_master(self, runner):
        result = runner.invoke(cli, ["info", "--master", "localhost:1"])
        assert result.exit_code != 0 or "error" in result.output.lower()
