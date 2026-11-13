{
  description = "Edge & local LLM quantization, KL-divergence evaluation, and hardware benchmark suite";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      inherit (nixpkgs) lib;
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = f: lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      formatter = forAllSystems (pkgs: pkgs.nixfmt-tree);

      packages = forAllSystems (
        pkgs:
        let
          space = pkgs.callPackage ./nix/space.nix { };
          hf-bench = pkgs.callPackage ./nix/cli.nix { };
        in
        {
          inherit space hf-bench;
          default = space;
        }
      );

      checks = forAllSystems (
        pkgs:
        let
          inherit (self.packages.${pkgs.stdenv.hostPlatform.system}) space hf-bench;
        in
        {
          default = space;
          smoke = space.tests.smoke;
          cli = hf-bench.tests.unit;
        }
      );

      devShells = forAllSystems (
        pkgs:
        let
          inherit (self.packages.${pkgs.stdenv.hostPlatform.system}) hf-bench;
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.nodejs
              pkgs.just
              hf-bench
            ];
          };
        }
      );
    };
}
