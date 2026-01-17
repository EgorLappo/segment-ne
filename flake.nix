{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    naersk = {
      url = "github:nix-community/naersk";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      naersk,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = (import nixpkgs) {
          inherit system;
        };

        naersk' = pkgs.callPackage naersk { };

        clib = with pkgs; [
          clang
          libclang
          openblas
        ];
      in
      rec {
        # For `nix build` & `nix run`:
        defaultPackage = naersk'.buildPackage {
          src = ./.;

          buildInputs = clib;
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath clib}";
        };

        # For `nix develop`:
        devShell = pkgs.mkShell {
          name = "rust-env";
          nativeBuildInputs = with pkgs; [
            rustc
            cargo
            rust-analyzer
            rustfmt
            clippy
            rustPlatform.bindgenHook
          ];

          buildInputs = clib;

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath clib}
          '';

        };
      }
    );
}
