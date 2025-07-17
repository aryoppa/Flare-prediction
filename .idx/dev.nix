nix
{pkgs, ...}: {
  channel = "stable-23.11";
  packages = [pkgs.vim pkgs.python3 pkgs.pip];
}