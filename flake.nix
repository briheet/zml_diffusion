{
  description = "ZML diffusion";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};

          bazelPackage =
            pkgs.callPackage "${nixpkgs}/pkgs/by-name/ba/bazel_9/build-support/bazelPackage.nix"
              { };

          bazel = (pkgs.bazel_9.override { version = "9.1.1"; }).overrideAttrs {
            src = pkgs.fetchzip {
              url = "https://github.com/bazelbuild/bazel/releases/download/9.1.1/bazel-9.1.1-dist.zip";
              hash = "sha256-NwZQcycUMAzos1wLdSlwv2EjhDcPVJgQTkLT57AjFvI=";
              stripRoot = false;
            };
          };

          dependencyHashes = {
            aarch64-darwin = "sha256-OkG1x7piehe4U6lqfeLFbtsBxs0jB95k8e08OYnbB30=";
          };

          zmlDiffusion = bazelPackage {
            name = "zml-diffusion";
            version = "0.1.0";

            src = self;

            inherit bazel;
            targets = [ "//:zml_diffusion" ];
            commandArgs = lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
              "--spawn_strategy=local"
            ];

            env = {
              GIT_SSL_CAINFO = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
              NIX_SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
              RULES_ZIG_CACHE_PREFIX = ".zig-cache";
              SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
              USE_BAZEL_VERSION = bazel.version;
            };

            nativeBuildInputs = with pkgs; [
              cacert
              gitMinimal
              gnupatch
              python3
            ];

            bazelRepoCacheFOD = {
              outputHash = dependencyHashes.${system} or lib.fakeHash;
              outputHashAlgo = "sha256";
            };

            installPhase = ''
              runHook preInstall

              mkdir -p "$out/bin"
              cp -L bazel-bin/zml_diffusion "$out/bin/zml_diffusion"

              if [[ -d bazel-bin/zml_diffusion.runfiles ]]; then
                cp -R -L \
                  bazel-bin/zml_diffusion.runfiles \
                  "$out/bin/zml_diffusion.runfiles"
              fi

              if [[ -f bazel-bin/zml_diffusion.runfiles_manifest ]]; then
                cp \
                  bazel-bin/zml_diffusion.runfiles_manifest \
                  "$out/bin/zml_diffusion.runfiles_manifest"
              fi

              runHook postInstall
            '';
          };
        in
        {
          default = zmlDiffusion;
        }
      );

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/zml_diffusion";
        };
      });

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShellNoCC {
            packages = with pkgs; [
              bazel-buildtools
              bazelisk
              cacert
              gitMinimal
              gnupatch
              nixfmt
              python3
            ];

            shellHook = ''
              echo "Bazel $(cat .bazelversion) · Zig 0.16.0 (managed by ZML)"
            '';
          };
        }
      );

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt);
    };
}
