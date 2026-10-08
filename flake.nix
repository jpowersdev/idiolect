{
  description = "Idiolect — a local voice cloning CLI";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-darwin" ];
      shell = system:
        let
          pkgs = import nixpkgs { inherit system; };
          inherit (pkgs.stdenv.hostPlatform) isLinux;
        in pkgs.mkShell ({
          # The same requirements as a manual install: Bun for the CLI, Python 3.12
          # and uv for the engine, and SoX for the microphone and speakers.
          packages = with pkgs; [ bun python312 uv sox ];
          IDIOLECT_NIX = "1";
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON = "${pkgs.python312}/bin/python3.12";
          SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
          PYTHONNOUSERSITE = "1";
        } // pkgs.lib.optionalAttrs isLinux {
          # PyPI wheels expect these at the usual FHS locations.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [ stdenv.cc.cc.lib zlib glib libsndfile ]);
        });
    in {
      devShells = nixpkgs.lib.genAttrs systems (system: { default = shell system; });
    };
}
