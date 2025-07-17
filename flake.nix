nix
{
  description = "A development environment with Python and Pip";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        buildInputs = [
          pkgs.python3
          pkgs.pip
        ];

        # Import the dev.nix settings if needed, or incorporate them here
        # import ./dev.nix { inherit pkgs; };
      };
    };
}